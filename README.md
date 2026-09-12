# Solar System Simulation (Pygame)

A small gravity simulation of the Sun and all eight planets, built with
**pygame**, **NumPy** and **SciPy**. It's based on the classic "Planet
Simulation in Python" tutorial, with better physics and a movable camera.

## Setup

```bash
pip install pygame numpy scipy
python main.py
```

Needs Python 3.9 or newer.

## Controls

| Key / mouse   | Action                                  |
|---------------|-----------------------------------------|
| Space         | Pause / resume                          |
| Up / Down     | Double / halve simulation speed         |
| Mouse wheel   | Zoom in or out around the cursor        |
| Left drag     | Pan the camera                          |
| 0 – 8         | Follow a body (0 = Sun, 3 = Earth, …)   |
| F             | Free camera (stop following)            |
| T             | Toggle orbit trails                     |
| L             | Toggle labels (distance from the Sun)   |
| R             | Reset                                   |
| Esc           | Quit                                    |

The outer planets are off-screen at first. Scroll out to see Jupiter through
Neptune.

## What's improved over the tutorial

| Tutorial                               | This version                                                   |
|----------------------------------------|----------------------------------------------------------------|
| Only the Sun ↔ planet pull is computed | Full N-body: every body attracts every other body              |
| Python loops over every pair           | Vectorised NumPy (all forces calculated in one step)           |
| Euler integration (orbits drift)       | Velocity Verlet: energy stays almost constant (drift shown on screen) |
| 1-day time step                        | 6-hour steps, several per frame, speed you can change          |
| Trails grow forever and slow it down   | Fixed-length trails (`deque(maxlen=...)`)                      |
| Fixed view, 4 inner planets            | Zoom, pan, follow any body, all 8 planets                      |
| Sun fixed at the centre                | The Sun moves too (total momentum starts at zero)              |
| Constants typed in by hand             | `G` and `AU` come from `scipy.constants`                       |

## Files

- `physics.py`: planet data, force calculation, Verlet step and energy. It
  doesn't use pygame, so you can import it and experiment on its own.
- `main.py`: the window, camera, input handling and drawing.

## Ideas to try

- Add a body to `BODIES` in `physics.py`, like the Moon or a comet on an
  eccentric orbit (use a slower starting speed).
- Make Jupiter 1000× heavier and watch the inner planets get thrown around.
- Change `DT` to a larger value and watch the energy drift go up.
