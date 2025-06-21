"""
Educational Physics Games using Pygame
Interactive games to help students understand physics concepts
"""

import numpy as np
import math
import random
import io
import base64
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

# Try to import pygame, with fallback for cloud environments
try:
    import pygame
    pygame.init()
    PYGAME_AVAILABLE = True
except (ImportError, pygame.error):
    PYGAME_AVAILABLE = False
    # Create a mock pygame module for cloud compatibility
    class MockPygame:
        def __init__(self):
            pass
        
        def Surface(self, size):
            return MockSurface(size)
        
        def Clock(self):
            return MockClock()
        
        def time(self):
            return self
        
        def get_size(self):
            return (800, 600)
        
        def tostring(self, surface, format):
            return b''
        
        class font:
            @staticmethod
            def Font(name, size):
                return MockFont()
        
        class image:
            @staticmethod
            def tostring(surface, format):
                return b''
        
        class draw:
            @staticmethod
            def circle(surface, color, pos, radius, width=0):
                pass
            
            @staticmethod
            def lines(surface, color, closed, points, width=1):
                pass
            
            @staticmethod
            def rect(surface, color, rect):
                pass
    
    class MockSurface:
        def __init__(self, size):
            self.size = size
        
        def fill(self, color):
            pass
        
        def blit(self, surface, pos):
            pass
        
        def get_size(self):
            return self.size
    
    class MockClock:
        def tick(self, fps):
            pass
    
    class MockFont:
        def render(self, text, antialias, color):
            return MockSurface((100, 20))
    
    pygame = MockPygame()

class GravityDropGame:
    """
    Gravity Basics Game: Students control projectile motion and gravity strength
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()
        
        # Physics parameters
        self.gravity = 9.81  # m/s²
        self.scale = 10  # pixels per meter
        self.dt = 0.02  # time step
        
        # Game objects
        self.balls = []
        self.targets = []
        self.score = 0
        self.level = 1
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        
        self.setup_level()
    
    def setup_level(self):
        """Setup targets and initial conditions for current level"""
        self.targets = []
        self.balls = []
        
        # Create targets based on level
        for i in range(self.level + 2):
            x = random.randint(100, self.width - 100)
            y = random.randint(self.height//2, self.height - 50)
            self.targets.append({'x': x, 'y': y, 'hit': False})
    
    def add_ball(self, start_x: int, start_y: int, velocity_x: float, velocity_y: float):
        """Add a new ball with given initial conditions"""
        ball = {
            'x': start_x,
            'y': start_y,
            'vx': velocity_x,
            'vy': velocity_y,
            'trail': [(start_x, start_y)]
        }
        self.balls.append(ball)
    
    def update_physics(self):
        """Update ball physics"""
        for ball in self.balls[:]:
            # Update velocity (gravity only affects y)
            ball['vy'] += self.gravity * self.scale * self.dt
            
            # Update position
            ball['x'] += ball['vx'] * self.dt
            ball['y'] += ball['vy'] * self.dt
            
            # Add to trail
            ball['trail'].append((ball['x'], ball['y']))
            if len(ball['trail']) > 50:
                ball['trail'].pop(0)
            
            # Check bounds
            if ball['y'] > self.height or ball['x'] < 0 or ball['x'] > self.width:
                self.balls.remove(ball)
                continue
            
            # Check target collisions
            for target in self.targets:
                if not target['hit']:
                    distance = math.sqrt((ball['x'] - target['x'])**2 + (ball['y'] - target['y'])**2)
                    if distance < 25:
                        target['hit'] = True
                        self.score += 10
                        break
    
    def draw(self):
        """Draw the game state"""
        self.screen.fill(self.BLACK)
        
        # Draw targets
        for target in self.targets:
            color = self.GREEN if target['hit'] else self.RED
            pygame.draw.circle(self.screen, color, (int(target['x']), int(target['y'])), 20)
        
        # Draw balls and trails
        for ball in self.balls:
            # Draw trail
            if len(ball['trail']) > 1:
                pygame.draw.lines(self.screen, self.YELLOW, False, ball['trail'], 2)
            
            # Draw ball
            pygame.draw.circle(self.screen, self.BLUE, (int(ball['x']), int(ball['y'])), 8)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        level_text = font.render(f"Level: {self.level}", True, self.WHITE)
        gravity_text = font.render(f"Gravity: {self.gravity:.1f} m/s²", True, self.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(level_text, (10, 50))
        self.screen.blit(gravity_text, (10, 90))
        
        return self.screen
    
    def get_game_state_image(self) -> str:
        """Get current game state as base64 encoded image"""
        surface = self.draw()
        
        # Convert pygame surface to PIL image
        w, h = surface.get_size()
        raw = pygame.image.tostring(surface, 'RGB')
        
        # Convert to base64
        import PIL.Image
        pil_image = PIL.Image.frombytes('RGB', (w, h), raw)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str


class OrbitDesignGame:
    """
    Orbital Mechanics Game: Students design stable orbits
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width, height))
        
        # Central body (planet/star)
        self.central_body = {
            'x': width // 2,
            'y': height // 2,
            'mass': 1e6,
            'radius': 30
        }
        
        # Satellites
        self.satellites = []
        self.target_orbits = []
        
        # Physics
        self.G = 6.67e-11 * 1e6  # Scaled gravitational constant
        self.dt = 0.1
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)  # Central body
        self.BLUE = (0, 100, 255)    # Satellites
        self.GREEN = (0, 255, 0)     # Target orbits
        self.RED = (255, 0, 0)       # Failed orbits
        
        self.score = 0
        self.setup_targets()
    
    def setup_targets(self):
        """Setup target orbital radii"""
        self.target_orbits = [150, 200, 250, 300]  # Target orbital radii
    
    def add_satellite(self, x: int, y: int, vx: float, vy: float):
        """Add a satellite with initial position and velocity"""
        satellite = {
            'x': float(x),
            'y': float(y),
            'vx': vx,
            'vy': vy,
            'trail': [(x, y)],
            'stable': False,
            'orbit_radius': 0
        }
        self.satellites.append(satellite)
    
    def update_physics(self):
        """Update orbital mechanics"""
        for sat in self.satellites[:]:
            # Calculate distance to central body
            dx = sat['x'] - self.central_body['x']
            dy = sat['y'] - self.central_body['y']
            r = math.sqrt(dx**2 + dy**2)
            
            if r < self.central_body['radius']:
                # Collision with central body
                self.satellites.remove(sat)
                continue
            
            # Gravitational force
            F = self.G * self.central_body['mass'] / r**2
            
            # Force components
            fx = -F * dx / r
            fy = -F * dy / r
            
            # Update velocity
            sat['vx'] += fx * self.dt
            sat['vy'] += fy * self.dt
            
            # Update position
            sat['x'] += sat['vx'] * self.dt
            sat['y'] += sat['vy'] * self.dt
            
            # Update trail
            sat['trail'].append((sat['x'], sat['y']))
            if len(sat['trail']) > 200:  # Longer trail for orbits
                sat['trail'].pop(0)
            
            # Check if orbit is stable
            sat['orbit_radius'] = r
            if len(sat['trail']) > 100:
                radii = [math.sqrt((p[0] - self.central_body['x'])**2 + 
                                 (p[1] - self.central_body['y'])**2) for p in sat['trail'][-50:]]
                if max(radii) - min(radii) < 20:  # Stable orbit criterion
                    sat['stable'] = True
                    
                    # Check if matches target orbit
                    avg_radius = sum(radii) / len(radii)
                    for target_r in self.target_orbits:
                        if abs(avg_radius - target_r) < 15:
                            self.score += 50
                            self.target_orbits.remove(target_r)
                            break
            
            # Remove if too far away
            if r > self.width:
                self.satellites.remove(sat)
    
    def draw(self):
        """Draw the orbital system"""
        self.screen.fill(self.BLACK)
        
        # Draw target orbit circles
        for radius in self.target_orbits:
            pygame.draw.circle(self.screen, self.GREEN, 
                             (self.central_body['x'], self.central_body['y']), 
                             int(radius), 2)
        
        # Draw central body
        pygame.draw.circle(self.screen, self.YELLOW,
                         (self.central_body['x'], self.central_body['y']),
                         self.central_body['radius'])
        
        # Draw satellites and trails
        for sat in self.satellites:
            # Draw trail
            if len(sat['trail']) > 1:
                color = self.GREEN if sat['stable'] else self.BLUE
                pygame.draw.lines(self.screen, color, False, 
                                [(int(p[0]), int(p[1])) for p in sat['trail']], 2)
            
            # Draw satellite
            color = self.GREEN if sat['stable'] else self.BLUE
            pygame.draw.circle(self.screen, color, (int(sat['x']), int(sat['y'])), 5)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        targets_text = font.render(f"Targets Left: {len(self.target_orbits)}", True, self.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(targets_text, (10, 50))
        
        return self.screen


class BlackHoleEscapeGame:
    """
    Black Hole Physics Game: Navigate around black hole avoiding tidal forces
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width, height))
        
        # Black hole
        self.black_hole = {
            'x': width // 2,
            'y': height // 2,
            'mass': 1e8,
            'schwarzschild_radius': 20,
            'event_horizon': 25
        }
        
        # Player spacecraft
        self.player = {
            'x': 100.0,
            'y': height // 2,
            'vx': 0.0,
            'vy': 0.0,
            'fuel': 100,
            'alive': True,
            'trail': []
        }
        
        # Collectibles (safe zones)
        self.collectibles = []
        
        # Physics
        self.G = 6.67e-11 * 1e8  # Scaled
        self.dt = 0.05
        self.c = 3e8  # Speed of light (scaled)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.PURPLE = (128, 0, 128)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        
        self.score = 0
        self.setup_collectibles()
    
    def setup_collectibles(self):
        """Setup collectible items"""
        for i in range(5):
            angle = i * 2 * math.pi / 5
            radius = 200 + random.randint(-50, 50)
            x = self.black_hole['x'] + radius * math.cos(angle)
            y = self.black_hole['y'] + radius * math.sin(angle)
            self.collectibles.append({
                'x': x, 'y': y, 'collected': False
            })
    
    def apply_thrust(self, thrust_x: float, thrust_y: float):
        """Apply thrust to player spacecraft"""
        if self.player['fuel'] > 0:
            self.player['vx'] += thrust_x * 0.5
            self.player['vy'] += thrust_y * 0.5
            self.player['fuel'] -= 1
    
    def update_physics(self):
        """Update black hole physics"""
        if not self.player['alive']:
            return
        
        # Distance to black hole
        dx = self.player['x'] - self.black_hole['x']
        dy = self.player['y'] - self.black_hole['y']
        r = math.sqrt(dx**2 + dy**2)
        
        # Check if past event horizon
        if r <= self.black_hole['event_horizon']:
            self.player['alive'] = False
            return
        
        # Gravitational acceleration (Newtonian approximation)
        a = self.G * self.black_hole['mass'] / r**2
        
        # Add relativistic effects near the black hole
        if r < 100:
            # Time dilation factor (simplified)
            time_factor = math.sqrt(1 - (2 * self.G * self.black_hole['mass']) / (r * self.c**2))
            a *= time_factor
        
        # Apply acceleration
        ax = -a * dx / r
        ay = -a * dy / r
        
        self.player['vx'] += ax * self.dt
        self.player['vy'] += ay * self.dt
        
        # Update position
        self.player['x'] += self.player['vx'] * self.dt
        self.player['y'] += self.player['vy'] * self.dt
        
        # Update trail
        self.player['trail'].append((self.player['x'], self.player['y']))
        if len(self.player['trail']) > 100:
            self.player['trail'].pop(0)
        
        # Check collectibles
        for item in self.collectibles:
            if not item['collected']:
                distance = math.sqrt((self.player['x'] - item['x'])**2 + 
                                   (self.player['y'] - item['y'])**2)
                if distance < 20:
                    item['collected'] = True
                    self.score += 20
                    self.player['fuel'] += 20  # Refuel bonus
        
        # Check bounds
        if (self.player['x'] < 0 or self.player['x'] > self.width or
            self.player['y'] < 0 or self.player['y'] > self.height):
            self.player['alive'] = False
    
    def draw(self):
        """Draw the black hole scenario"""
        self.screen.fill(self.BLACK)
        
        # Draw tidal force visualization (concentric circles)
        for i in range(5, 15):
            radius = i * 20
            alpha = max(0, 255 - radius * 2)
            color = (alpha//3, 0, alpha//2)
            if alpha > 10:
                pygame.draw.circle(self.screen, color,
                                 (self.black_hole['x'], self.black_hole['y']),
                                 radius, 1)
        
        # Draw event horizon
        pygame.draw.circle(self.screen, self.RED,
                         (self.black_hole['x'], self.black_hole['y']),
                         self.black_hole['event_horizon'], 3)
        
        # Draw black hole
        pygame.draw.circle(self.screen, self.BLACK,
                         (self.black_hole['x'], self.black_hole['y']),
                         self.black_hole['schwarzschild_radius'])
        
        # Draw collectibles
        for item in self.collectibles:
            if not item['collected']:
                pygame.draw.circle(self.screen, self.YELLOW,
                                 (int(item['x']), int(item['y'])), 8)
        
        # Draw player trail
        if len(self.player['trail']) > 1:
            color = self.GREEN if self.player['alive'] else self.RED
            pygame.draw.lines(self.screen, color, False,
                            [(int(p[0]), int(p[1])) for p in self.player['trail']], 2)
        
        # Draw player
        if self.player['alive']:
            pygame.draw.circle(self.screen, self.BLUE,
                             (int(self.player['x']), int(self.player['y'])), 6)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        fuel_text = font.render(f"Fuel: {int(self.player['fuel'])}", True, self.WHITE)
        status_text = font.render("ALIVE" if self.player['alive'] else "SPAGHETTIFIED!", 
                                True, self.GREEN if self.player['alive'] else self.RED)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(fuel_text, (10, 50))
        self.screen.blit(status_text, (10, 90))
        
        return self.screen


class GravitationalWaveGame:
    """
    Gravitational Waves Game: Detect wave patterns and match frequencies
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width, height))
        
        # Wave parameters
        self.waves = []
        self.detectors = []
        self.time = 0
        self.score = 0
        self.target_frequency = None
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        
        self.setup_detectors()
        self.generate_wave()
    
    def setup_detectors(self):
        """Setup gravitational wave detectors"""
        detector_positions = [(200, 300), (400, 300), (600, 300)]
        for i, pos in enumerate(detector_positions):
            self.detectors.append({
                'x': pos[0],
                'y': pos[1],
                'signal': [],
                'active': False,
                'name': f'LIGO-{i+1}'
            })
    
    def generate_wave(self):
        """Generate a new gravitational wave event"""
        # Binary merger parameters
        frequency = random.uniform(50, 500)  # Hz
        amplitude = random.uniform(0.5, 2.0)
        chirp_rate = random.uniform(1.1, 2.0)  # Frequency increases
        
        self.waves.append({
            'frequency': frequency,
            'amplitude': amplitude,
            'chirp_rate': chirp_rate,
            'start_time': self.time,
            'duration': 2.0,  # seconds
            'detected': False
        })
        
        self.target_frequency = frequency
    
    def update_wave_physics(self, dt: float):
        """Update gravitational wave propagation"""
        self.time += dt
        
        # Update detector signals
        for detector in self.detectors:
            signal_value = 0
            
            for wave in self.waves:
                wave_age = self.time - wave['start_time']
                if 0 <= wave_age <= wave['duration']:
                    # Calculate distance effect (simplified)
                    distance = math.sqrt((detector['x'] - self.width//2)**2 + 
                                       (detector['y'] - self.height//2)**2)
                    distance_factor = 1 / (1 + distance / 100)
                    
                    # Chirp signal (frequency increase over time)
                    current_freq = wave['frequency'] * (wave['chirp_rate'] ** wave_age)
                    
                    # Calculate strain (simplified)
                    strain = (wave['amplitude'] * distance_factor * 
                             math.sin(2 * math.pi * current_freq * wave_age) *
                             math.exp(-wave_age))  # Decay
                    
                    signal_value += strain
            
            # Add noise
            signal_value += random.uniform(-0.1, 0.1)
            
            detector['signal'].append(signal_value)
            if len(detector['signal']) > 200:
                detector['signal'].pop(0)
    
    def activate_detector(self, detector_index: int):
        """Activate a detector for analysis"""
        if 0 <= detector_index < len(self.detectors):
            self.detectors[detector_index]['active'] = not self.detectors[detector_index]['active']
    
    def analyze_signals(self):
        """Analyze detector signals for wave detection"""
        active_detectors = [d for d in self.detectors if d['active']]
        
        if len(active_detectors) >= 2:
            # Check for coincident detection
            for wave in self.waves:
                if not wave['detected']:
                    # Simple coincidence check
                    strong_signals = 0
                    for detector in active_detectors:
                        if len(detector['signal']) > 50:
                            max_signal = max(abs(s) for s in detector['signal'][-50:])
                            if max_signal > 0.5:
                                strong_signals += 1
                    
                    if strong_signals >= 2:
                        wave['detected'] = True
                        self.score += 100
                        # Generate new wave
                        self.generate_wave()
    
    def draw(self):
        """Draw the gravitational wave detection interface"""
        self.screen.fill(self.BLACK)
        
        # Draw wave source (binary system)
        center_x, center_y = self.width // 2, 100
        for i in range(2):
            angle = self.time * 2 + i * math.pi
            x = center_x + 30 * math.cos(angle)
            y = center_y + 30 * math.sin(angle)
            pygame.draw.circle(self.screen, self.YELLOW, (int(x), int(y)), 8)
        
        # Draw wave propagation (concentric circles)
        wave_radius = (self.time * 100) % 400
        if wave_radius > 50:
            pygame.draw.circle(self.screen, self.BLUE, (center_x, center_y), 
                             int(wave_radius), 2)
        
        # Draw detectors
        for i, detector in enumerate(self.detectors):
            color = self.GREEN if detector['active'] else self.WHITE
            pygame.draw.rect(self.screen, color,
                           (detector['x']-20, detector['y']-10, 40, 20))
            
            # Draw detector signal
            if len(detector['signal']) > 1:
                points = []
                for j, signal in enumerate(detector['signal'][-100:]):
                    x = detector['x'] - 50 + j
                    y = detector['y'] + 50 + signal * 50
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, color, False, points, 2)
            
            # Label
            font = pygame.font.Font(None, 24)
            label = font.render(detector['name'], True, self.WHITE)
            self.screen.blit(label, (detector['x']-30, detector['y']) + 30)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        time_text = font.render(f"Time: {self.time:.1f}s", True, self.WHITE)
        
        if self.target_frequency:
            freq_text = font.render(f"Target: {self.target_frequency:.1f} Hz", True, self.WHITE)
            self.screen.blit(freq_text, (10, 90))
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 50))
        
        # Instructions
        inst_font = pygame.font.Font(None, 24)
        instructions = [
            "Click detectors to activate",
            "Need 2+ active for detection",
            "Look for coincident signals"
        ]
        for i, instruction in enumerate(instructions):
            text = inst_font.render(instruction, True, self.WHITE)
            self.screen.blit(text, (self.width - 250, 10 + i * 25))
        
        return self.screen


class GameManager:
    """
    Manager class to handle all educational games
    """
    
    def __init__(self):
        self.games = {
            'gravity_drop': GravityDropGame(),
            'orbit_design': OrbitDesignGame(),
            'black_hole_escape': BlackHoleEscapeGame(),
            'gravitational_waves': GravitationalWaveGame()
        }
        self.current_game = None
    
    def get_game(self, game_name: str):
        """Get a specific game instance"""
        return self.games.get(game_name)
    
    def run_game_step(self, game_name: str, action: Dict = None) -> str:
        """Run one step of a game and return the state as base64 image"""
        game = self.games.get(game_name)
        if not game:
            return None
        
        # Apply action if provided
        if action and game_name == 'gravity_drop':
            if action.get('launch_ball'):
                game.add_ball(action['x'], action['y'], action['vx'], action['vy'])
            if action.get('change_gravity'):
                game.gravity = action['gravity']
        
        elif action and game_name == 'orbit_design':
            if action.get('add_satellite'):
                game.add_satellite(action['x'], action['y'], action['vx'], action['vy'])
        
        elif action and game_name == 'black_hole_escape':
            if action.get('thrust'):
                game.apply_thrust(action['thrust_x'], action['thrust_y'])
        
        elif action and game_name == 'gravitational_waves':
            if action.get('activate_detector') is not None:
                game.activate_detector(action['activate_detector'])
        
        # Update physics
        if hasattr(game, 'update_physics'):
            game.update_physics()
        if hasattr(game, 'update_wave_physics'):
            game.update_wave_physics(0.05)
        if hasattr(game, 'analyze_signals'):
            game.analyze_signals()
        
        # Get visual state
        surface = game.draw()
        
        # Convert to base64
        w, h = surface.get_size()
        raw = pygame.image.tostring(surface, 'RGB')
        
        import PIL.Image
        pil_image = PIL.Image.frombytes('RGB', (w, h), raw)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def get_game_info(self, game_name: str) -> Dict:
        """Get information about a game"""
        info = {
            'gravity_drop': {
                'name': 'Gravity Drop Challenge',
                'description': 'Master projectile motion by launching balls to hit targets. Adjust gravity and launch parameters!',
                'controls': 'Set launch angle, velocity, and gravity strength',
                'learning_goals': ['Projectile motion', 'Gravity effects', 'Trajectory prediction']
            },
            'orbit_design': {
                'name': 'Orbital Designer',
                'description': 'Design stable orbits around a central body. Learn about orbital mechanics!',
                'controls': 'Click to place satellites with different velocities',
                'learning_goals': ['Orbital mechanics', 'Gravitational forces', 'Stable vs unstable orbits']
            },
            'black_hole_escape': {
                'name': 'Black Hole Escape',
                'description': 'Navigate around a black hole, avoid the event horizon, and collect items!',
                'controls': 'Use thrust carefully - fuel is limited!',
                'learning_goals': ['Black hole physics', 'Event horizons', 'Relativistic effects']
            },
            'gravitational_waves': {
                'name': 'Wave Detective',
                'description': 'Detect gravitational waves from merging black holes using multiple detectors!',
                'controls': 'Activate detectors and look for coincident signals',
                'learning_goals': ['Gravitational waves', 'Signal detection', 'Scientific method']
            }
        }
        return info.get(game_name, {})


# Utility functions for Streamlit integration
def create_game_interface(game_name: str) -> str:
    """Create an HTML interface for a game"""
    game_info = GameManager().get_game_info(game_name)
    
    html = f"""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f9f9f9;">
        <h3 style="color: #2E7D32; margin-top: 0;">{game_info.get('name', 'Educational Game')}</h3>
        <p><strong>Description:</strong> {game_info.get('description', 'Interactive physics game')}</p>
        <p><strong>Controls:</strong> {game_info.get('controls', 'Use mouse and keyboard')}</p>
        <p><strong>Learning Goals:</strong></p>
        <ul>
    """
    
    for goal in game_info.get('learning_goals', []):
        html += f"<li>{goal}</li>"
    
    html += """
        </ul>
        <div style="background-color: #E8F5E8; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <p style="margin: 0; font-style: italic; color: #2E7D32;">
                🎮 Interactive game will load below. Use the controls to explore physics concepts!
            </p>
        </div>
    </div>
    """
    
    return html


if __name__ == "__main__":
    # Test the games
    manager = GameManager()
    
    # Test gravity drop game
    game_state = manager.run_game_step('gravity_drop', {
        'launch_ball': True,
        'x': 100, 'y': 100, 'vx': 150, 'vy': -50
    })
    
    print("Educational games module loaded successfully!")
    print(f"Available games: {list(manager.games.keys())}")
