"""
main.py - Pygame front end for the solar system simulation.

Run with:  python main.py

Controls
  Space          pause / resume
  Up / Down      speed up / slow down time
  Mouse wheel    zoom (towards the cursor)
  Left drag      pan the camera
  0 - 8          follow a body (0 = Sun, 3 = Earth, ...)
  F              stop following (free camera)
  T              toggle orbit trails
  L              toggle labels
  R              reset the simulation
  Esc            quit
"""

from collections import deque

import numpy as np
import pygame

import physics
from physics import AU, DAY, BODIES


WIDTH, HEIGHT = 1000, 800
FPS = 60
DT = DAY / 4            
TRAIL_LENGTH = 600      
TRAIL_EVERY = 4         

BACKGROUND = (8, 8, 20)
WHITE = (240, 240, 240)
GREY = (140, 140, 160)


class Camera:
    """Converts between world coordinates (metres) and screen pixels."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scale = 250 / AU           
        self.center = np.zeros(2)        
        self.follow = 0                  

    def to_screen(self, world):
        """World (N, 2) array -> screen (N, 2) array. y is flipped so +y is up."""
        rel = (world - self.center) * self.scale
        return np.column_stack((WIDTH / 2 + rel[:, 0], HEIGHT / 2 - rel[:, 1]))

    def to_world(self, screen_xy):
        x, y = screen_xy
        return self.center + np.array([x - WIDTH / 2, HEIGHT / 2 - y]) / self.scale

    def zoom(self, factor, mouse_pos):
        """Zoom while keeping the point under the mouse fixed."""
        before = self.to_world(mouse_pos)
        self.scale *= factor
        after = self.to_world(mouse_pos)
        if self.follow is None:         
            self.center += before - after


class Simulation:
    """Holds the physical state plus the trail history for drawing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos, self.vel, self.masses = physics.initial_state()
        self.acc = physics.accelerations(self.pos, self.masses)
        self.time = 0.0
        self.steps = 0
        self.start_energy = physics.total_energy(self.pos, self.vel, self.masses)
        self.trails = [deque(maxlen=TRAIL_LENGTH) for _ in BODIES]

    def advance(self, n_steps):
        """Run n_steps physics steps, recording trail points as we go."""
        for _ in range(n_steps):
            self.pos, self.vel, self.acc = physics.step(
                self.pos, self.vel, self.acc, self.masses, DT)
            self.time += DT
            self.steps += 1
            if self.steps % TRAIL_EVERY == 0:
                for trail, p in zip(self.trails, self.pos):
                    trail.append(p.copy())

    def energy_drift(self):
        """Relative change in total energy since start (accuracy check)."""
        e = physics.total_energy(self.pos, self.vel, self.masses)
        return abs((e - self.start_energy) / self.start_energy)


def draw(screen, font, sim, cam, show_trails, show_labels):
    screen.fill(BACKGROUND)

  
    if show_trails:
        for trail, body in zip(sim.trails, BODIES):
            if len(trail) > 1:
                pts = cam.to_screen(np.array(trail))
                dim = tuple(c // 2 for c in body[5])      
                pygame.draw.lines(screen, dim, False, pts.tolist(), 1)

    
    screen_pos = cam.to_screen(sim.pos)
    sun_pos = sim.pos[0]
    for i, (body, (x, y)) in enumerate(zip(BODIES, screen_pos)):
        name, _, _, _, radius, colour = body
        pygame.draw.circle(screen, colour, (int(x), int(y)), radius)
        if show_labels and i > 0:
            dist_au = np.linalg.norm(sim.pos[i] - sun_pos) / AU
            text = font.render(f"{name}  {dist_au:.2f} AU", True, GREY)
            screen.blit(text, (x + radius + 4, y - 8))


def draw_hud(screen, font, sim, cam, steps_per_frame, paused, clock):
    """Text overlay in the top-left corner."""
    follow = BODIES[cam.follow][0] if cam.follow is not None else "free"
    lines = [
        f"Day {sim.time / DAY:,.0f}   (year {sim.time / DAY / 365.25:.2f})",
        f"Speed: {steps_per_frame * DT / DAY:g} days/frame" + ("   PAUSED" if paused else ""),
        f"Zoom: {cam.scale * AU:.1f} px/AU   Following: {follow}",
        f"Energy drift: {sim.energy_drift():.2e}   FPS: {clock.get_fps():.0f}",
        "Space pause | Up/Down speed | Wheel zoom | Drag pan | 0-8 follow | T/L/R",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, WHITE if i < 4 else GREY), (10, 10 + i * 18))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Solar System Simulation")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    sim, cam = Simulation(), Camera()
    steps_per_frame = 4            
    paused, show_trails, show_labels = False, True, True
    dragging = False

    running = True
    while running:
        clock.tick(FPS)

 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame * 2, 256)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame // 2, 1)
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_f:
                    cam.follow = None
                elif event.key == pygame.K_r:
                    sim.reset()
                    cam.reset()
                elif pygame.K_0 <= event.key <= pygame.K_8:
                    cam.follow = event.key - pygame.K_0

            elif event.type == pygame.MOUSEWHEEL:
                cam.zoom(1.15 if event.y > 0 else 1 / 1.15, pygame.mouse.get_pos())

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                cam.follow = None          
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam.center -= np.array([dx, -dy]) / cam.scale

       
        if not paused:
            sim.advance(steps_per_frame)
        if cam.follow is not None:
            cam.center = sim.pos[cam.follow].copy()

       
        draw(screen, font, sim, cam, show_trails, show_labels)
        draw_hud(screen, font, sim, cam, steps_per_frame, paused, clock)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
