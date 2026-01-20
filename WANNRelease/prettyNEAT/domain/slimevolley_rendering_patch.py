"""
Rendering patch for slimevolleygym to work without gym.envs.classic_control.rendering

This module provides a complete pygame-based implementation of the gym rendering API
to make slimevolleygym's native rendering work properly.
"""

import numpy as np
import pygame
import math


class Transform:
    """Transformation class for positioning and rotating geometries."""
    
    def __init__(self):
        self.translation = (0.0, 0.0)
        self.rotation = 0.0
        self.scale = (1.0, 1.0)
    
    def set_translation(self, x, y):
        self.translation = (x, y)
    
    def set_rotation(self, angle):
        self.rotation = angle
    
    def set_scale(self, sx, sy):
        self.scale = (sx, sy)


class Geom:
    """Base geometry class."""
    
    def __init__(self):
        self.color = (255, 255, 255, 255)
        self.attrs = []
    
    def set_color(self, r, g, b, alpha=1.0):
        self.color = (int(r * 255), int(g * 255), int(b * 255), int(alpha * 255))
    
    def add_attr(self, attr):
        self.attrs.append(attr)
    
    def render(self, surface):
        """Override in subclasses."""
        pass
    
    def _apply_transforms(self, points):
        """Apply all transformations to points."""
        result = np.array(points, dtype=float)
        
        for attr in self.attrs:
            if isinstance(attr, Transform):
                # Apply scale
                result[:, 0] *= attr.scale[0]
                result[:, 1] *= attr.scale[1]
                
                # Apply rotation
                if attr.rotation != 0:
                    cos_r = math.cos(attr.rotation)
                    sin_r = math.sin(attr.rotation)
                    x = result[:, 0] * cos_r - result[:, 1] * sin_r
                    y = result[:, 0] * sin_r + result[:, 1] * cos_r
                    result[:, 0] = x
                    result[:, 1] = y
                
                # Apply translation
                result[:, 0] += attr.translation[0]
                result[:, 1] += attr.translation[1]
        
        return result


class FilledPolygon(Geom):
    """Filled polygon geometry."""
    
    def __init__(self, vertices):
        super().__init__()
        self.vertices = np.array(vertices, dtype=float)
    
    def render(self, surface):
        """Render the polygon on the surface."""
        # Apply transformations
        points = self._apply_transforms(self.vertices)
        
        # Convert to pygame coordinates (flip Y axis)
        height = surface.get_height()
        pygame_points = [(int(p[0]), int(height - p[1])) for p in points]
        
        # Draw the polygon
        if len(pygame_points) >= 3:
            pygame.draw.polygon(surface, self.color[:3], pygame_points)


class PolyLine(Geom):
    """Polyline geometry."""
    
    def __init__(self, vertices, close=False):
        super().__init__()
        self.vertices = np.array(vertices, dtype=float)
        self.close = close
    
    def set_linewidth(self, width):
        self.linewidth = width
    
    def render(self, surface):
        """Render the polyline on the surface."""
        # Apply transformations
        points = self._apply_transforms(self.vertices)
        
        # Convert to pygame coordinates
        height = surface.get_height()
        pygame_points = [(int(p[0]), int(height - p[1])) for p in points]
        
        # Draw the line
        if len(pygame_points) >= 2:
            pygame.draw.lines(surface, self.color[:3], self.close, pygame_points, 2)


class Circle(Geom):
    """Circle geometry."""
    
    def __init__(self, radius, res=30):
        super().__init__()
        self.radius = radius
        self.res = res
    
    def render(self, surface):
        """Render the circle on the surface."""
        # Get center position from transforms
        center = np.array([[0.0, 0.0]])
        transformed = self._apply_transforms(center)
        
        # Convert to pygame coordinates
        height = surface.get_height()
        x = int(transformed[0, 0])
        y = int(height - transformed[0, 1])
        
        # Get radius (affected by scale)
        radius = self.radius
        for attr in self.attrs:
            if isinstance(attr, Transform):
                radius *= attr.scale[0]  # Use x scale
        
        # Draw the circle
        pygame.draw.circle(surface, self.color[:3], (x, y), int(radius))


class Viewer:
    """
    Pygame-based viewer that implements the gym rendering API.
    """
    
    def __init__(self, width, height):
        """Initialize viewer with given dimensions."""
        self.width = width
        self.height = height
        self.window = None
        self.clock = None
        self.isopen = False
        self.geoms = []
        self.onetime_geoms = []
    
    def render(self, return_rgb_array=False):
        """
        Render the environment.
        
        Args:
            return_rgb_array: If True, return RGB array instead of displaying
            
        Returns:
            RGB array if return_rgb_array is True, else None
        """
        if not self.isopen:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("SlimeVolley")
            self.clock = pygame.time.Clock()
            self.isopen = True
        
        # Clear screen
        self.window.fill((0, 0, 0))
        
        # Draw all geometries
        for geom in self.geoms:
            geom.render(self.window)
        for geom in self.onetime_geoms:
            geom.render(self.window)
        
        # Clear onetime geometries
        self.onetime_geoms = []
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        if return_rgb_array:
            # Get pixel array from window
            rgb_array = pygame.surfarray.array3d(self.window)
            # Convert from (width, height, 3) to (height, width, 3)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            return rgb_array
        else:
            return None
    
    def add_geom(self, geom):
        """Add a geometry to be rendered."""
        self.geoms.append(geom)
    
    def add_onetime(self, geom):
        """Add a geometry to be rendered once."""
        self.onetime_geoms.append(geom)
    
    def close(self):
        """Close the viewer."""
        if self.isopen:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    
    def __del__(self):
        """Cleanup."""
        self.close()


class SimpleImageViewer:
    """
    Simple pygame-based image viewer for pixel rendering mode.
    """
    
    def __init__(self, maxwidth=2160):
        self.window = None
        self.clock = None
        self.maxwidth = maxwidth
        self.isopen = False
    
    def imshow(self, arr):
        """
        Display an image array.
        
        Args:
            arr: numpy array of shape (H, W, 3) with values 0-255
        """
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.isopen = True
        
        arr = np.asarray(arr, dtype=np.uint8)
        
        # Ensure RGB format
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=2)
        
        # Get dimensions
        height, width = arr.shape[:2]
        
        # Scale if necessary
        if width > self.maxwidth:
            scale = self.maxwidth / width
            width = int(width * scale)
            height = int(height * scale)
        
        # Create or resize window
        if self.window is None or self.window.get_size() != (width, height):
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("SlimeVolley")
            self.clock = pygame.time.Clock()
        
        # Convert array to pygame surface
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        
        # Scale if needed
        if surf.get_size() != (width, height):
            surf = pygame.transform.scale(surf, (width, height))
        
        # Draw to screen
        self.window.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        # Control frame rate
        self.clock.tick(60)
    
    def close(self):
        """Close the viewer window."""
        if self.isopen:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.window = None
    
    def __del__(self):
        """Cleanup."""
        self.close()


# Factory functions for creating geometries (gym rendering API)

def make_circle(radius, res=30, filled=True):
    """Create a circle geometry."""
    return Circle(radius, res)


def make_polygon(vertices, filled=True):
    """Create a polygon geometry."""
    if filled:
        return FilledPolygon(vertices)
    else:
        return PolyLine(vertices, close=True)


def make_polyline(vertices):
    """Create a polyline geometry."""
    return PolyLine(vertices, close=False)


def make_capsule(length, width):
    """Create a capsule (rounded rectangle) geometry."""
    # Approximate with a polygon
    l, r, t, b = 0, length, width/2, -width/2
    return FilledPolygon([(l,b), (l,t), (r,t), (r,b)])


def make_half_circle(radius=10, res=20, filled=True):
    """Create a half circle (semicircle) geometry."""
    points = []
    for i in range(res+1):
        ang = math.pi - math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, close=True)


def apply_rendering_patch():
    """
    Monkeypatch slimevolleygym to use our pygame-based rendering module
    instead of the missing gym.envs.classic_control.rendering module.
    """
    import slimevolleygym.slimevolley as sv
    from types import ModuleType
    
    # Create a fake rendering module with all the necessary classes and functions
    rendering_module = ModuleType('rendering')
    rendering_module.Viewer = Viewer
    rendering_module.SimpleImageViewer = SimpleImageViewer
    rendering_module.Geom = Geom
    rendering_module.Transform = Transform
    rendering_module.FilledPolygon = FilledPolygon
    rendering_module.PolyLine = PolyLine
    rendering_module.Circle = Circle
    rendering_module.make_circle = make_circle
    rendering_module.make_polygon = make_polygon
    rendering_module.make_polyline = make_polyline
    rendering_module.make_capsule = make_capsule
    rendering_module.make_half_circle = make_half_circle
    
    # Monkeypatch the checkRendering function
    def patched_checkRendering():
        """Patched version that uses our rendering module."""
        sv.rendering = rendering_module
    
    # Replace the checkRendering function
    sv.checkRendering = patched_checkRendering
    
    # Initialize rendering
    sv.rendering = None
    
    return True
