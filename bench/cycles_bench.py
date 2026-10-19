"""Deterministic Cycles/EEVEE benchmark scene for comparing GPUs and render settings.

    blender -b --factory-startup -noaudio -P cycles_bench.py -- \
        --engine CYCLES --device OPTIX --frames 48 --spp 64 --out /tmp/bench [--no-persistent] [--lanes-tag x]

Prints one JSON line prefixed with BENCH= at the end: first-frame time (kernel load + BVH), steady-state
seconds per frame, and the settings used, so runs on different boxes can be compared directly.
"""

import argparse
import json
import math
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
p.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
p.add_argument("--frames", type=int, default=48)
p.add_argument("--spp", type=int, default=64)
p.add_argument("--res", default="1080x1920")
p.add_argument("--out", default="/tmp/bench")
p.add_argument("--no-persistent", action="store_true")
p.add_argument("--png-compression", type=int, default=15)
p.add_argument("--tag", default="")
a = p.parse_args(argv)

scene = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)

# A mid-weight scene: displaced ground, 60 subdivided objects mixing diffuse, glossy and glass, three area lights
# and a sky. Seeded layout so every box renders the same image.
import random

rng = random.Random(7)
bpy.ops.mesh.primitive_plane_add(size=60)
ground = bpy.context.object
mod = ground.modifiers.new("sub", "SUBSURF")
mod.subdivision_type = "SIMPLE"
mod.levels = mod.render_levels = 6
tex = bpy.data.textures.new("n", "CLOUDS")
disp = ground.modifiers.new("disp", "DISPLACE")
disp.texture = tex
disp.strength = 0.6


def material(name, kind, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    if kind == "glossy":
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.15
    elif kind == "glass":
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.02
    else:
        bsdf.inputs["Roughness"].default_value = 0.6
    return m


mats = [material(f"m{i}", k, (rng.random(), rng.random(), rng.random()))
        for i, k in enumerate(["diffuse", "glossy", "glass"] * 4)]
for i in range(60):
    if i % 2:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=rng.uniform(0.4, 1.2), segments=48, ring_count=24)
    else:
        bpy.ops.mesh.primitive_monkey_add(size=rng.uniform(0.8, 1.6))
        bpy.ops.object.modifier_add(type="SUBSURF")
        bpy.context.object.modifiers[-1].levels = bpy.context.object.modifiers[-1].render_levels = 2
    ob = bpy.context.object
    ob.location = (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(0.5, 2.0))
    ob.rotation_euler = (0, 0, rng.uniform(0, math.tau))
    ob.data.materials.append(mats[i % len(mats)])
    # A third of the objects move, so the BVH is rebuilt every frame like in a real animation.
    if i % 3 == 0:
        ob.keyframe_insert("location", frame=1)
        ob.location.z += 1.5
        ob.keyframe_insert("location", frame=a.frames)

for i, (x, y) in enumerate([(-6, -6), (6, -4), (0, 8)]):
    bpy.ops.object.light_add(type="AREA", location=(x, y, 8))
    bpy.context.object.data.energy = 1500
    bpy.context.object.data.size = 4

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
sky = world.node_tree.nodes.new("ShaderNodeTexSky")
world.node_tree.links.new(sky.outputs[0], world.node_tree.nodes["Background"].inputs[0])

bpy.ops.object.camera_add(location=(0, -22, 9))
cam = bpy.context.object
scene.camera = cam
cam.rotation_euler = (math.radians(68), 0, 0)
cam.keyframe_insert("location", frame=1)
cam.location.x += 6
cam.keyframe_insert("location", frame=a.frames)

w, h = (int(v) for v in a.res.split("x"))
r = scene.render
r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
r.fps = 30
scene.frame_start, scene.frame_end = 1, a.frames
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
r.image_settings.compression = a.png_compression
r.filepath = a.out.rstrip("/") + "/f_####"
r.use_persistent_data = not a.no_persistent

device_used = "n/a"
if a.engine == "CYCLES":
    r.engine = "CYCLES"
    c = scene.cycles
    if a.device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = a.device
        prefs.refresh_devices()
        gpus = [d for d in prefs.devices if d.type == a.device]
        for d in prefs.devices:
            d.use = d.type == a.device
        if not gpus:
            print(f"BENCH_ERROR=no {a.device} device", flush=True)
            sys.exit(3)
        c.device = "GPU"
        device_used = ",".join(d.name for d in gpus)
    c.samples = a.spp
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_use_gpu = True
    c.use_light_tree = False
else:
    # 4.2-4.x call EEVEE Next BLENDER_EEVEE_NEXT; 5.x renamed it back to BLENDER_EEVEE.
    for name in (a.engine, "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            r.engine = name
            break
        except TypeError:
            continue
    scene.eevee.taa_render_samples = 32

times = []
_t = [0.0]


def pre(_s, _d=None):
    _t[0] = time.perf_counter()


def post(_s, _d=None):
    times.append(time.perf_counter() - _t[0])


bpy.app.handlers.render_pre.append(pre)
bpy.app.handlers.render_post.append(post)

t0 = time.perf_counter()
bpy.ops.render.render(animation=True)
wall = time.perf_counter() - t0
steady = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else None
print("BENCH=" + json.dumps({
    "tag": a.tag, "engine": a.engine, "device": a.device, "gpus": device_used, "frames": a.frames,
    "spp": a.spp, "res": a.res, "persistent": not a.no_persistent, "first_s": round(times[0], 2) if times else None,
    "median_s": round(steady, 3) if steady else None, "wall_s": round(wall, 1),
    "blender": bpy.app.version_string,
}), flush=True)
