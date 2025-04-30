from flask import Flask, render_template, Response, jsonify, request
import numpy as np
from flybody.fly_envs import flight_imitation
import cv2
import threading
import queue
import json
import logging
import time
import os
import sys
import traceback
from functools import wraps
import mujoco
import importlib.util
import platform
from dm_control import _render

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for simulation state
frame_queue = queue.Queue(maxsize=10)
control_queue = queue.Queue(maxsize=10)
env = None
simulation_active = False
simulation_thread = None
simulation_error = None
env_lock = threading.Lock()  # Lock for environment access

def handle_errors(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }), 500
    return wrapped

def test_basic_mujoco_env():
    """Test if we can create and run a basic MuJoCo environment."""
    logger.info("\nTesting basic MuJoCo environment...")
    try:
        from dm_control import suite
        logger.info("Creating basic cartpole environment...")
        env = suite.cartpole.swingup()
        
        # Test physics
        logger.info("Testing physics access...")
        if not hasattr(env, 'physics'):
            raise RuntimeError("Basic environment missing physics")
        if not hasattr(env.physics, 'model'):
            raise RuntimeError("Basic environment physics missing model")
        if env.physics.model is None:
            raise RuntimeError("Basic environment model is None")
            
        # Test rendering
        logger.info("Testing basic environment rendering...")
        pixels = env.physics.render(camera_id=0)
        if pixels is None or pixels.size == 0:
            raise RuntimeError("Basic environment render failed")
        logger.info(f"Basic environment render successful, shape: {pixels.shape}")
        
        # Test step
        logger.info("Testing basic environment step...")
        timestep = env.reset()
        action = env.action_spec().generate_value()
        timestep = env.step(action)
        
        logger.info("Basic MuJoCo environment test successful")
        return True
        
    except Exception as e:
        logger.error(f"Basic MuJoCo environment test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def verify_mujoco_installation():
    """Verify MuJoCo installation and configuration."""
    logger.info("\nVerifying MuJoCo installation...")
    
    # Check MuJoCo
    try:
        import mujoco
        logger.info(f"MuJoCo version: {mujoco.__version__}")
        logger.info(f"MuJoCo path: {mujoco.__file__}")
    except ImportError as e:
        logger.error(f"Failed to import MuJoCo: {str(e)}")
        return False
        
    # Check dm_control
    try:
        import dm_control
        logger.info(f"dm_control version: {dm_control.__version__}")
        logger.info(f"dm_control path: {dm_control.__file__}")
    except ImportError as e:
        logger.error(f"Failed to import dm_control: {str(e)}")
        return False
    
    # Test basic MuJoCo functionality
    try:
        logger.info("Testing basic MuJoCo model creation...")
        xml = """
        <mujoco>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        logger.info("Basic MuJoCo model creation successful")
    except Exception as e:
        logger.error(f"Failed to create basic MuJoCo model: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
    # Test basic environment creation
    if not test_basic_mujoco_env():
        return False
    
    return True

def verify_flybody_installation():
    """Verify flybody package installation and dependencies."""
    logger.info("\nVerifying flybody installation...")
    
    try:
        # Check if flybody is importable
        import flybody
        logger.info(f"flybody package found at: {flybody.__file__}")
        
        # Check fly_envs module
        from flybody import fly_envs
        logger.info("fly_envs module found")
        
        # Check flight_imitation environment
        if not hasattr(fly_envs, 'flight_imitation'):
            raise ImportError("flight_imitation environment not found in fly_envs")
        logger.info("flight_imitation environment found")
        
        return True
    except Exception as e:
        logger.error(f"flybody verification failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_diagnostic_checks():
    """Run comprehensive diagnostic checks for MuJoCo and dependencies."""
    logger.info("Running diagnostic checks...")
    
    # Check Python environment
    logger.info("\nPython Environment:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python architecture: {platform.architecture()}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Check required packages
    packages_to_check = ['mujoco', 'dm_control', 'numpy', 'cv2']
    logger.info("\nChecking required packages:")
    for package in packages_to_check:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'version not found')
            file_location = getattr(module, '__file__', 'location not found')
            logger.info(f"{package}: version={version}, location={file_location}")
        except ImportError as e:
            logger.error(f"Failed to import {package}: {str(e)}")
            return False
    
    # Check MuJoCo specific configuration
    logger.info("\nMuJoCo Configuration:")
    try:
        import mujoco
        logger.info(f"MuJoCo version: {mujoco.__version__}")
        logger.info(f"MuJoCo path: {mujoco.__file__}")
        
        # Try to create a basic MuJoCo model
        logger.info("Testing basic MuJoCo model creation...")
        try:
            model = mujoco.MjModel.from_xml_string('<mujoco/>')
            logger.info("Basic MuJoCo model creation successful")
        except Exception as e:
            logger.error(f"Failed to create basic MuJoCo model: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"MuJoCo configuration error: {str(e)}")
        return False
    
    # Verify flybody installation
    if not verify_flybody_installation():
        return False
    
    # Check environment variables
    logger.info("\nEnvironment Variables:")
    important_vars = ['MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', 'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']
    for var in important_vars:
        value = os.environ.get(var, 'not set')
        logger.info(f"{var}: {value}")
    
    return True

class EnvironmentWrapper:
    """Thread-safe wrapper for environment with proper context handling."""
    def __init__(self):
        self.env = None
        self._render_context = None
        self._render_lock = threading.Lock()
        self._env_lock = threading.Lock()
        self.render_width = 1280  # Increased from 640
        self.render_height = 720  # Increased from 480
        
    def initialize(self):
        """Initialize the environment in the current thread."""
        with self._env_lock:
            if self.env is not None:
                try:
                    self.env.close()
                except:
                    pass
                self.env = None
            
            logger.info("Creating flight_imitation environment...")
            try:
                self.env = flight_imitation(
                    disable_legs=True,
                    force_actuators=True,
                    terminal_com_dist=float('inf')
                )
                
                # Verify environment components
                if not hasattr(self.env, 'physics'):
                    raise RuntimeError("Environment missing physics attribute")
                if not hasattr(self.env.physics, 'model'):
                    raise RuntimeError("Physics missing model attribute")
                if self.env.physics.model is None:
                    raise RuntimeError("Physics model is None")
                    
                # Test action space
                action_spec = self.env.action_spec()
                logger.info(f"Action space shape: {action_spec.shape}")
                logger.info(f"Action space bounds: min={action_spec.minimum}, max={action_spec.maximum}")
                
                return True
            except Exception as e:
                logger.error(f"Environment initialization failed: {str(e)}")
                logger.error(traceback.format_exc())
                self.env = None
                return False
    
    def render(self):
        """Thread-safe rendering with proper context management."""
        with self._render_lock:
            if self.env is None:
                raise RuntimeError("Environment not initialized")
                
            if self._render_context is None:
                if os.environ.get('MUJOCO_GL') == 'glfw':
                    self._render_context = _render.Renderer(max_width=self.render_width, max_height=self.render_height)
            
            try:
                if self._render_context is not None:
                    with self._render_context.make_current():
                        return self.env.physics.render(camera_id=0, width=self.render_width, height=self.render_height)
                else:
                    return self.env.physics.render(camera_id=0, width=self.render_width, height=self.render_height)
            except Exception as e:
                logger.error(f"Render error: {str(e)}")
                logger.error(traceback.format_exc())
                raise
    
    def step(self, action):
        """Thread-safe environment stepping."""
        with self._env_lock:
            if self.env is None:
                raise RuntimeError("Environment not initialized")
            return self.env.step(action)
    
    def action_spec(self):
        """Get action specification."""
        with self._env_lock:
            if self.env is None:
                raise RuntimeError("Environment not initialized")
            return self.env.action_spec()
    
    def close(self):
        """Clean up resources."""
        with self._env_lock:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception as e:
                    logger.error(f"Error closing environment: {str(e)}")
                self.env = None
            
        with self._render_lock:
            if self._render_context is not None:
                try:
                    self._render_context.free()
                except Exception as e:
                    logger.error(f"Error freeing render context: {str(e)}")
                self._render_context = None

def try_initialize_with_backend(backend):
    """Try to initialize environment with a specific backend."""
    logger.info(f"\nAttempting initialization with {backend} backend...")
    
    # Store original environment variables
    original_gl = os.environ.get('MUJOCO_GL')
    original_device = os.environ.get('MUJOCO_EGL_DEVICE_ID')
    
    try:
        # Set backend
        os.environ['MUJOCO_GL'] = backend
        if backend == 'egl':
            os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
        
        logger.info(f"Using renderer: {backend}")
        logger.info(f"Current environment variables:")
        for var in ['MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', 'LD_LIBRARY_PATH', 'PATH']:
            logger.info(f"{var}: {os.environ.get(var, 'not set')}")
        
        # First test if we can create a basic environment with this backend
        if not test_basic_mujoco_env():
            raise RuntimeError(f"Basic environment test failed with {backend} backend")
        
        logger.info("Creating flight_imitation environment...")
        try:
            base_env = flight_imitation(
                disable_legs=True,
                force_actuators=True,
                terminal_com_dist=float('inf')
            )
        except Exception as create_error:
            logger.error(f"Error creating environment: {str(create_error)}")
            logger.error(traceback.format_exc())
            raise
        
        if base_env is None:
            raise RuntimeError("Environment creation returned None")
        
        # Verify environment components step by step
        logger.info("Verifying environment components...")
        
        if not hasattr(base_env, 'physics'):
            raise RuntimeError("Environment missing physics attribute")
        logger.info("Physics attribute found")
        
        if not hasattr(base_env.physics, 'model'):
            raise RuntimeError("Physics missing model attribute")
        logger.info("Model attribute found")
        
        if base_env.physics.model is None:
            raise RuntimeError("Physics model is None")
        logger.info("Model is valid")
        
        # Create environment wrapper
        logger.info("Creating environment wrapper...")
        new_env = EnvironmentWrapper()
        
        # Test render
        logger.info("Testing render...")
        try:
            test_frame = new_env.render()
            if test_frame is None:
                raise RuntimeError("Render returned None")
            if test_frame.size == 0:
                raise RuntimeError("Render returned empty frame")
            logger.info(f"Render test successful - frame shape: {test_frame.shape}")
        except Exception as render_error:
            logger.error(f"Render test failed: {str(render_error)}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info(f"Successfully initialized with {backend} backend")
        return new_env
        
    except Exception as e:
        logger.error(f"Failed with {backend} backend: {str(e)}")
        logger.error(traceback.format_exc())
        return None
        
    finally:
        # Restore original environment variables
        if original_gl is not None:
            os.environ['MUJOCO_GL'] = original_gl
        elif 'MUJOCO_GL' in os.environ:
            del os.environ['MUJOCO_GL']
            
        if original_device is not None:
            os.environ['MUJOCO_EGL_DEVICE_ID'] = original_device
        elif 'MUJOCO_EGL_DEVICE_ID' in os.environ:
            del os.environ['MUJOCO_EGL_DEVICE_ID']

def initialize_environment():
    """Initialize the fly environment."""
    global env, simulation_error
    logger.info("Starting environment initialization...")
    
    try:
        # Run diagnostic checks first
        logger.info("Running diagnostic checks...")
        if not run_diagnostic_checks():
            raise RuntimeError("Diagnostic checks failed - please check logs for details")
        
        # Try different backends in order of preference
        backends = ['glfw', 'egl', 'osmesa']
        new_env = None
        
        for backend in backends:
            new_env = try_initialize_with_backend(backend)
            if new_env is not None:
                break
        
        if new_env is None:
            raise RuntimeError("Failed to initialize with any backend - check logs for details")
        
        # Log action space information
        try:
            action_spec = new_env.env.action_spec()
            logger.info(f"Action space shape: {action_spec.shape}")
            logger.info(f"Action space bounds: min={action_spec.minimum}, max={action_spec.maximum}")
        except Exception as e:
            logger.error(f"Error getting action space: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        simulation_error = None
        return new_env
        
    except Exception as e:
        error_msg = f"Environment initialization failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        simulation_error = error_msg
        raise

def simulation_loop():
    """Main simulation loop that runs in a separate thread."""
    global simulation_active, env, simulation_error
    
    logger.debug("Starting simulation loop")
    try:
        # Initialize environment in the simulation thread
        with env_lock:
            if env is None:
                env = EnvironmentWrapper()
            
            if not env.initialize():
                simulation_error = "Failed to initialize environment"
                simulation_active = False
                return
            
    except Exception as e:
        simulation_error = f"Failed to initialize environment in simulation thread: {e}"
        logger.error(simulation_error)
        logger.error(traceback.format_exc())
        simulation_active = False
        return
    
    frame_count = 0
    last_fps_check = time.time()
    last_food_time = time.time()
    last_sleep_time = time.time()
    last_move_time = time.time()
    food_interval = 10  # seconds
    sleep_interval = 15  # seconds
    move_interval = 1   # seconds
    hunger_threshold = 60
    sleepiness_threshold = 60
    # For demonstration, we keep hunger/sleepiness as backend variables
    fly_hunger = 0
    fly_sleepiness = 0

    while simulation_active:
        try:
            current_time = time.time()
            if current_time - last_fps_check >= 1.0:
                logger.debug(f"Simulation FPS: {frame_count / (current_time - last_fps_check):.1f}")
                frame_count = 0
                last_fps_check = current_time
            
            # --- Autonomous food and sleep logic ---
            # Increase hunger and sleepiness over time
            fly_hunger = min(100, fly_hunger + 0.3 * (current_time - last_fps_check))
            fly_sleepiness = min(100, fly_sleepiness + 0.2 * (current_time - last_fps_check))

            # Add food if hungry
            if fly_hunger > hunger_threshold and (current_time - last_food_time) > food_interval:
                logger.debug("[AUTO] Adding food for fly (hunger high)")
                fly_hunger = max(0, fly_hunger - 30)
                last_food_time = current_time
            # Add sleep spot if sleepy
            if fly_sleepiness > sleepiness_threshold and (current_time - last_sleep_time) > sleep_interval:
                logger.debug("[AUTO] Adding sleep spot for fly (sleepiness high)")
                fly_sleepiness = max(0, fly_sleepiness - 40)
                last_sleep_time = current_time

            # Get latest control input (non-blocking)
            try:
                control = control_queue.get_nowait()
                logger.debug(f"Processing control input: {control[:5]}...")
            except queue.Empty:
                try:
                    # Autonomous movement: random walk every move_interval
                    if (current_time - last_move_time) > move_interval:
                        action_spec = env.action_spec()
                        control = np.random.uniform(action_spec.minimum, action_spec.maximum)
                        last_move_time = current_time
                        logger.debug(f"[AUTO] Sending random movement control: {control[:5]}")
                    else:
                        control = np.zeros(env.action_spec().shape[0])
                except Exception as e:
                    logger.error(f"Error getting action spec: {str(e)}")
                    simulation_active = False
                    break
            
            # Step the simulation and render
            try:
                timestep = env.step(control)
                frame_count += 1
                
                # Render frame
                frame = env.render()
                
                # Convert to JPEG for streaming
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_queue.put_nowait(jpeg.tobytes())
                
            except Exception as e:
                simulation_error = f"Error in simulation step: {e}"
                logger.error(simulation_error)
                logger.error(traceback.format_exc())
                simulation_active = False
                break
                
        except Exception as e:
            simulation_error = f"Critical error in simulation loop: {e}"
            logger.error(simulation_error)
            logger.error(traceback.format_exc())
            simulation_active = False
            break
    
    logger.debug(f"Simulation loop ended. Error: {simulation_error}")
    # Clean up environment
    with env_lock:
        if env is not None:
            env.close()
            env = None

def generate_frames():
    """Generator function for streaming frames."""
    logger.debug("Starting frame generation")
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        try:
            logger.debug("Waiting for frame...")
            frame = frame_queue.get(timeout=1.0)  # Reduced timeout for faster feedback
            current_time = time.time()
            frame_count += 1
            
            if frame_count % 30 == 0:  # Log every 30 frames
                fps = 30 / (current_time - last_frame_time)
                logger.debug(f"Streaming at {fps:.1f} FPS")
                last_frame_time = current_time
                
            logger.debug(f"Got frame #{frame_count}, size: {len(frame)} bytes")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            logger.warning("Frame queue empty, yielding empty frame")
            # Generate an empty frame or placeholder
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty_frame, 'Waiting for simulation...', (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', empty_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Serve the main page."""
    logger.debug("Serving index page")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming video frames."""
    logger.debug("Starting video feed")
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@handle_errors
def get_status():
    """Get the current simulation status."""
    global simulation_active, simulation_thread, simulation_error, env
    
    status = {
        'initialized': env is not None,
        'running': simulation_active and simulation_thread and simulation_thread.is_alive(),
        'error': simulation_error if simulation_error else None
    }
    return jsonify(status)

@app.route('/start')
@handle_errors
def start_simulation():
    """Start the simulation."""
    global simulation_active, simulation_thread, simulation_error, env
    
    logger.debug("Received start simulation request")
    
    if simulation_thread and simulation_thread.is_alive():
        return jsonify({"status": "error", "message": "Simulation already running"})
    
    # Clear any previous error
    simulation_error = None
    
    # Create new environment wrapper (but don't initialize yet)
    with env_lock:
        if env is not None:
            env.close()
        env = EnvironmentWrapper()
    
    # Clear queues
    while not frame_queue.empty():
        frame_queue.get()
    while not control_queue.empty():
        control_queue.get()
    
    simulation_active = True
    simulation_thread = threading.Thread(target=simulation_loop)
    simulation_thread.daemon = True
    simulation_thread.start()
    logger.debug("Simulation thread started")
    
    # Wait for environment initialization
    max_wait = 30  # Maximum wait time in seconds
    start_time = time.time()
    while simulation_active and time.time() - start_time < max_wait:
        if simulation_error:  # Check for initialization error
            break
        if env and env.env is not None:  # Check if environment is initialized
            break
        time.sleep(0.1)
    
    if not simulation_active or simulation_error:
        error_msg = simulation_error or "Environment initialization timed out"
        logger.error(f"Start failed: {error_msg}")
        return jsonify({"status": "error", "message": error_msg})
    
    try:
        action_spec = env.action_spec()
        return jsonify({
            "status": "started",
            "action_space": {
                "shape": action_spec.shape[0],
                "minimum": action_spec.minimum.tolist() if hasattr(action_spec.minimum, 'tolist') else float(action_spec.minimum),
                "maximum": action_spec.maximum.tolist() if hasattr(action_spec.maximum, 'tolist') else float(action_spec.maximum)
            }
        })
    except Exception as e:
        simulation_active = False
        error_msg = f"Error getting action space: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_msg})

@app.route('/stop')
@handle_errors
def stop_simulation():
    """Stop the simulation."""
    global simulation_active, env
    logger.debug("Received stop simulation request")
    simulation_active = False
    
    # Clean up environment
    with env_lock:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
            env = None
    
    return jsonify({"status": "stopped"})

@app.route('/control', methods=['POST'])
@handle_errors
def update_control():
    """Update control inputs."""
    global simulation_active, simulation_thread, simulation_error, env
    
    try:
        with env_lock:
            if env is None:
                return jsonify({"status": "error", "message": "Environment not initialized"})
            
            if not simulation_active or not simulation_thread or not simulation_thread.is_alive():
                error_msg = simulation_error or "Simulation not running"
                logger.warning(f"Control update rejected: {error_msg}")
                return jsonify({"status": "error", "message": error_msg})
            
            control_data = request.get_json()
            if not isinstance(control_data, list):
                raise ValueError("Control data must be a list")
            
            # Get action spec from environment
            try:
                action_spec = env.env.action_spec()
                expected_size = action_spec.shape[0]
                
                if len(control_data) != expected_size:
                    error_msg = f"Control array size mismatch. Expected {expected_size}, got {len(control_data)}"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg})
                
                # Convert control data to numpy array
                control = np.array(control_data, dtype=np.float32)
                
                # Handle array-like bounds
                minimum = action_spec.minimum
                maximum = action_spec.maximum
                if not np.isscalar(minimum):
                    minimum = np.broadcast_to(minimum, control.shape)
                if not np.isscalar(maximum):
                    maximum = np.broadcast_to(maximum, control.shape)
                
                # Validate and clip control values
                if not np.all(np.logical_and(control >= minimum, control <= maximum)):
                    logger.warning("Control values out of bounds, clipping to valid range")
                    control = np.clip(control, minimum, maximum)
                
                logger.debug(f"Received control update: first 5 values = {control[:5]}")
                control_queue.put(control)
                return jsonify({"status": "success"})
                
            except Exception as e:
                error_msg = f"Error validating control input: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Control validation error traceback: {traceback.format_exc()}")
                return jsonify({"status": "error", "message": error_msg})
            
    except Exception as e:
        logger.error(f"Error updating control: {str(e)}")
        logger.error(f"Control update error traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/update_camera', methods=['POST'])
@handle_errors
def update_camera():
    """Update camera position and orientation."""
    global env
    
    try:
        with env_lock:
            if env is None:
                return jsonify({"status": "error", "message": "Environment not initialized"})
            
            camera_data = request.get_json()
            if not isinstance(camera_data, dict):
                raise ValueError("Camera data must be a dictionary")
            
            # Extract camera parameters
            rotation = camera_data.get('rotation', {'x': 0, 'y': 0, 'z': 0})
            zoom = camera_data.get('zoom', 1.0)
            pan = camera_data.get('pan', {'x': 0, 'y': 0})
            
            # Update camera in MuJoCo
            try:
                # Scale down the zoom factor for better default view
                adjusted_zoom = 0.5 + (zoom * 0.5)  # This makes default zoom=1 give a reasonable view
                env.env.physics.model.cam_pos[0] = [
                    4.0 * adjusted_zoom * np.cos(rotation['y']) * np.cos(rotation['x']),
                    4.0 * adjusted_zoom * np.sin(rotation['y']) * np.cos(rotation['x']),
                    4.0 * adjusted_zoom * np.sin(rotation['x'])
                ]
                env.env.physics.model.cam_quat[0] = [1, 0, 0, 0]  # Reset camera orientation
                
                # Apply pan offset with scaled sensitivity
                env.env.physics.model.cam_pos[0][0] += pan['x'] * adjusted_zoom
                env.env.physics.model.cam_pos[0][1] += pan['y'] * adjusted_zoom
                
                return jsonify({"status": "success"})
                
            except Exception as e:
                logger.error(f"Error updating camera: {str(e)}")
                return jsonify({"status": "error", "message": str(e)})
            
    except Exception as e:
        logger.error(f"Error in camera update: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, threaded=True) 