import os
import re
import json
import time
import uuid
import base64
import logging
import subprocess
import traceback
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from openai import OpenAI

# ========================= CONFIG =========================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Suppress Flask/Werkzeug request logs (keep only pipeline logs)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent
BLENDER_EXEC = os.environ.get("BLENDER_EXEC", "/home/abdullah/blender/blender-5.0.0-linux-x64/blender")
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

MAX_ERROR_RETRIES = int(os.environ.get("MAX_ERROR_RETRIES", 999))  # NO LIMIT - retry until success

# Pipeline progress tracking (in-memory, per session)
pipeline_progress = {}  # session_id -> { "stage": str, "detail": str, "attempt": int, "max_attempts": int }

def update_progress(session_id, stage, detail="", attempt=0, max_attempts=MAX_ERROR_RETRIES):
    pipeline_progress[session_id] = {
        "stage": stage,
        "detail": detail,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "timestamp": datetime.now().isoformat()
    }
    log(f"[PROGRESS] {session_id}: {stage} — {detail}", "PROGRESS")

# OpenRouter (for ALL models: Claude, Gemini, Codex, GPT Vision)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
) if OPENROUTER_API_KEY else None

# Model identifiers
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.1-pro-preview")
CODEX_MODEL = os.environ.get("CODEX_MODEL", "openai/gpt-5.3-codex")
GPT_VISION_MODEL = os.environ.get("GPT_VISION_MODEL", "openai/gpt-4o")
CLAUDE_OPUS_MODEL = os.environ.get("CLAUDE_OPUS_MODEL", "anthropic/claude-opus-4-6")
CLAUDE_SONNET_MODEL = os.environ.get("CLAUDE_SONNET_MODEL", "anthropic/claude-sonnet-4-6")

# ========================= LOGGING =========================

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level}] {msg}")

# ========================= SYSTEM PROMPT (loaded from master_prompt.txt) =========================

_prompt_path = PROJECT_ROOT / "master_prompt.txt"
if _prompt_path.exists():
    SYSTEM_PROMPT = _prompt_path.read_text()
    log(f"Loaded master prompt: {len(SYSTEM_PROMPT)} chars, {SYSTEM_PROMPT.count(chr(10))} lines")
else:
    raise FileNotFoundError(f"master_prompt.txt not found at {_prompt_path}")

_part_prompt_path = PROJECT_ROOT / "part_regen_prompt.txt"
PART_REGEN_PROMPT = _part_prompt_path.read_text() if _part_prompt_path.exists() else ""
if PART_REGEN_PROMPT:
    log(f"Loaded part regen prompt: {len(PART_REGEN_PROMPT)} chars")

# ========================= PROMPT BUILDERS =========================

def build_sketch_generation_prompt(user_description=""):
    """Build a prompt for image-to-3D generation. The image is sent separately via the LLM API."""
    base = """You are looking at a REFERENCE IMAGE of a ring/jewelry piece.
Your job is to write a Blender Python script that builds 3D geometry matching this image as closely as possible.

Study the image carefully. Identify:
- The overall shape (ring, bangle, cuff, etc.)
- Every stone, flower, vine, filigree, or decorative element — count them exactly
- The band profile and proportions
- How elements connect and are positioned

Then write the COMPLETE Blender script to replicate it.

RULES:
- Replicate EXACTLY what you see — same element count, same shapes, same proportions
- Cabochon stones = smooth domes, NOT faceted. Faceted stones = brilliant-cut geometry
- If the image shows an open/cuff style with a gap: make the opening VERY SMALL — only 2-4% of the full circle (~7-14 degrees). Real cuffs have tiny openings. Even if the photo angle makes the gap look wide, the 3D model gap must be narrow
- Match stone setting types exactly (bezel, prong, flush, channel, pave — whatever the image shows)
- Build every decorative detail visible in the image"""

    if user_description:
        base += f"""

ADDITIONAL USER NOTES: "{user_description}"
Use these notes as extra context. If they conflict with the image, prioritize the IMAGE."""

    base += """

TECHNICAL REMINDERS:
- The head/setting grows directly from the band's top — they are ONE connected piece.
- Gems sit INSIDE their settings, not floating. Prongs grip the gem, not pass through it.
- All build_* functions share the same dimension variables so parts line up.
- Use modifiers generously (Bevel, Subsurf, etc.) for quality.
- No materials, no cameras, no lights. Output ONLY geometry code."""

    return base


def build_edit_prompt(original_code, edit_request):
    return f"""Here is the COMPLETE current ring script:

```python
{original_code}
```

The user wants this change: "{edit_request}"

CRITICAL RULES:
1. Return the COMPLETE updated script — every function, every import, every line.
2. Change ONLY what the user requested. Everything else must be BYTE-FOR-BYTE IDENTICAL.
3. Do NOT rename functions, reorder code, change comments, or "improve" unrelated parts.
4. Keep ALL engineering rules: bmesh only, no bpy.ops.mesh, no materials, nuke+build pattern.
5. ZERO materials, ZERO lighting, ZERO export code.
6. USE modifiers (BEVEL, SUBSURF) for metal quality.
7. Return ONLY Python code. No explanations. No markdown fences."""


def build_smart_edit_prompt(full_code, edit_request, func_name):
    return f"""Here is the COMPLETE ring script. The user wants to modify ONLY the function `{func_name}`.

```python
{full_code}
```

User's edit request: "{edit_request}"

CRITICAL RULES:
1. Return the COMPLETE script — ALL functions, ALL imports, ALL code.
2. Modify ONLY the function `{func_name}`. Every other function, variable, import, and line of code must remain BYTE-FOR-BYTE IDENTICAL.
3. Keep the same function signature: def {func_name}(...)
4. The modified function must still work with all other functions (shared constants, same return type).
5. Use ONLY bmesh (no bpy.ops.mesh, no bpy.ops.transform). NO materials, NO lighting.
6. Return ONLY Python code. No explanations. No markdown fences."""


def build_part_regen_prompt(full_code, part_type, user_description):
    return f"""Here is the COMPLETE existing ring script:

```python
{full_code}
```

The user wants to REGENERATE the "{part_type}" part of this ring.
User's description: "{user_description}"

{PART_REGEN_PROMPT}

CRITICAL:
1. Return the COMPLETE script with the {part_type} function(s) COMPLETELY REWRITTEN.
2. ALL OTHER functions must remain BYTE-FOR-BYTE IDENTICAL.
3. The new {part_type} must use the SAME shared dimension variables.
4. The new {part_type} must connect properly to adjacent parts.
5. Make the new {part_type} aesthetically superior and structurally perfect.
6. ONLY bmesh geometry. NO materials, NO lighting.
7. Return ONLY Python code. No explanations. No markdown fences."""


def build_fix_prompt(code, error_text, spatial_report=None):
    base_prompt = f"""═══════════════════════════════════════════════════════════════════════════════
BLENDER SCRIPT CRASH — MANDATORY ONE-ATTEMPT FIX
═══════════════════════════════════════════════════════════════════════════════

This Blender 5.0 Python script CRASHED. You have ONE attempt to fix it. If your
fix fails, the entire pipeline fails permanently. There are NO second chances.solve all the syntax geometry
and all erors in one attempt while keep the design aesthatically correct

CRASHED SCRIPT:
```python
{code}
```

CRASH ERROR:
```
{error_text}
```
"""

    if spatial_report:
        base_prompt += f"""
SPATIAL CONTEXT (mesh positions/bounds from previous attempt):
{spatial_report[:3000]}
"""

    base_prompt += """
═══════════════════════════════════════════════════════════════════════════════
PHASE 1: FIX THE REPORTED CRASH (read the traceback, find the exact line)
═══════════════════════════════════════════════════════════════════════════════

Read the traceback above. Identify the EXACT line number and root cause.
Fix that specific error first.

═══════════════════════════════════════════════════════════════════════════════
PHASE 2: FULL SCRIPT AUDIT — FIX EVERY OTHER HIDDEN ERROR
═══════════════════════════════════════════════════════════════════════════════

After fixing the reported crash, you MUST scan the ENTIRE script line-by-line
for ALL of the following crash patterns. Fix EVERY one you find. Do NOT skip
any category — scripts that pass Phase 1 but crash on a hidden error are
UNACCEPTABLE.

─── AUDIT 1: NameError — UNDEFINED VARIABLES AND FUNCTIONS ──────────────────

Trace EVERY variable name and function call in the script. For EACH one, verify
it is either imported, defined above its first use, or passed as a parameter.

MANDATORY CHECKS:
  □ Is the main function named exactly build() (not build_ring, main, generate)?
  □ Is ring_col created INSIDE build() with:
        ring_col = bpy.data.collections.new("Ring")
        bpy.context.scene.collection.children.link(ring_col)
    BEFORE any helper function uses it?
  □ Is ring_col passed as a parameter to EVERY helper function that links objects?
  □ Are ALL math functions imported? Check for: sin, cos, tan, asin, acos, atan2,
    pi, radians, degrees, sqrt, floor, ceil — if ANY is used, it MUST be imported.
  □ Are ALL custom utility functions (lerp, smoothstep, ease_in_out, clamp,
    normalize_vec, cross_vec, simple_noise, _safe_face, bm_to_object, etc.)
    DEFINED in the script BEFORE their first call?
  □ For EVERY function call in the script: does that function exist? Is it spelled
    correctly? Does it have the right number of arguments?

─── AUDIT 2: BLENDER 5.0 BANNED API — INSTANT CRASH IF PRESENT ─────────────

Search the ENTIRE script for these patterns. If found, apply the fix shown:

  ❌ mesh.use_auto_smooth = True          → DELETE the entire line
  ❌ mesh.use_auto_smooth = False         → DELETE the entire line
  ❌ mesh.auto_smooth_angle = ...         → DELETE the entire line
  ❌ face.use_smooth = True (on BMFace)   → DELETE. Set on mesh.polygons AFTER bm.to_mesh()
  ❌ material.shadow_method = ...         → DELETE (no materials allowed at all)
  ❌ material.blend_method = ...          → DELETE (no materials allowed at all)
  ❌ bool_mod.solver = 'FAST'            → CHANGE to 'FLOAT'
  ❌ bm.recalc_normals()                 → CHANGE to bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  ❌ bmesh.ops.create_cylinder(...)      → CHANGE to bmesh.ops.create_cone(bm, radius1=r, radius2=r, ...)
  ❌ diameter1= / diameter2=              → CHANGE to radius1= / radius2=
  ❌ Matrix.Identity()                    → CHANGE to Matrix.Identity(4)
  ❌ bmesh.ops.bevel(vertex_only=...)    → DELETE vertex_only kwarg (invalid in 5.0)
  ❌ bpy.ops.mesh.primitive_*            → REWRITE using bmesh (no bpy.ops for geometry)
  ❌ bpy.ops.object.mode_set(...)        → DELETE (not needed in headless mode)
  ❌ bpy.ops.transform.*                 → REWRITE using direct coordinate manipulation
  ❌ bpy.ops.mesh.*                      → REWRITE using bmesh operations
  ❌ bpy.ops.object.select_all(...)      → DELETE (not needed)
  ❌ bpy.ops.object.join(...)            → DELETE (objects must stay separate)
  ❌ Any material/shader/node/light code  → DELETE entirely
  ❌ bpy.context.copy() for override     → DELETE (removed in 4.0+)
  ❌ from mathutils import noise         → DELETE. Use hash-based noise instead.
  ❌ Euler(angles, 'LOCAL')              → CHANGE to Euler(angles, 'XYZ')
  ❌ sub.quality = ...                   → WRAP in try/except: pass

─── AUDIT 3: BMESH CRASHES ──────────────────────────────────────────────────

  □ ensure_lookup_table(): Search for EVERY bm.verts[i], bm.edges[i], bm.faces[i]
    index access. Before EACH such access, verify that ensure_lookup_table() was
    called AFTER the last vertex/edge/face addition. If not, ADD IT:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

  □ Face creation: EVERY bm.faces.new() call MUST be wrapped in try/except:
        try:
            bm.faces.new(face_verts)
        except (ValueError, IndexError):
            pass

  □ Duplicate vertices: Before creating a face, verify the vertex list has no
    duplicates. If a list might have dupes, deduplicate:
        unique = list(dict.fromkeys(vert_list))
        if len(unique) >= 3:
            try: bm.faces.new(unique)
            except: pass

  □ BMesh after free: NEVER access bm.verts, bm.faces, or any BMesh element
    reference after calling bm.free(). Extract all needed data BEFORE freeing:
        coords = [(v.co.x, v.co.y, v.co.z) for v in bm.verts]
        bm.to_mesh(mesh)
        bm.free()

  □ bridge_loops: If bmesh.ops.bridge_loops is used, verify it receives exactly
    two separate edge loops with MATCHING vertex counts. If uncertain, replace
    with manual face creation using bm.faces.new().

─── AUDIT 4: COLLECTION AND CONTEXT CRASHES ─────────────────────────────────

  □ NEVER use bpy.context.collection.objects.link(obj) — after scene clear it
    can be None. ALWAYS use: ring_col.objects.link(obj)

  □ NEVER access collections by name: bpy.data.collections["Ring_Assembly"]
    This raises KeyError. ALWAYS create fresh: bpy.data.collections.new("Ring")

  □ Scene cleanup MUST use list():
        for obj in list(bpy.data.objects): bpy.data.objects.remove(obj, do_unlink=True)
        for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)
    NEVER iterate bpy.data.* directly while removing — wrap in list() first.

  □ NEVER call bpy.ops.object.modifier_apply() on objects with shared mesh data.
    If multiple objects share a mesh, modifier_apply crashes with "Cannot apply
    to a multi user". SOLUTION: Don't apply modifiers — leave them on the stack.

─── AUDIT 5: MATH AND TYPE ERRORS ──────────────────────────────────────────

  □ Math domain errors: Search for asin(), acos(), sqrt(). Clamp ALL inputs:
        asin(max(-1.0, min(1.0, val)))
        acos(max(-1.0, min(1.0, val)))
        sqrt(max(0.0, val))

  □ NoneType arithmetic: Trace EVERY helper function. Verify EVERY code path
    returns a numeric value. If a function has if/elif without else, ADD an
    else with a default return. If a function might return None, add a
    safety return at the end.

  □ Division by zero: Search for every / and // operator. If the divisor could
    ever be zero, add a guard: x / max(divisor, 1e-10)

  □ Type mismatches: Verify no tuple.cross() (use mathutils Vector), no
    Vector.z on 2D vectors (check len(vec) >= 3 first).

  □ Unpacking errors: If a function returns a tuple and the caller unpacks it
    (x, y, z = func()), verify the function ALWAYS returns the expected number
    of values on ALL code paths.

─── AUDIT 6: SYNTAX AND STRUCTURE ──────────────────────────────────────────

  □ Bracket matching: Verify EVERY ( has ), EVERY [ has ], EVERY { has }.
    Scan for unclosed brackets especially in long expressions, f-strings, and
    multi-line function calls.

  □ Indentation: Verify consistent 4-space indentation. No mixed tabs/spaces.

  □ Complete code: Every function body must be complete. No truncated code,
    no "..." placeholders, no "# TODO" comments, no pass-only functions that
    should have real code.

  □ String literals: Verify all strings are properly closed. No unterminated
    f-strings or triple-quoted strings.

  □ Import completeness: ALL of these must be present at the top:
        import bpy
        import bmesh
        from math import (sin, cos, tan, asin, acos, atan2,
                          pi, radians, degrees, sqrt, floor, ceil)
        from mathutils import Vector, Matrix
    If the script uses random, hashlib, or other stdlib modules, import them too.

─── AUDIT 7: MODIFIER SAFETY ───────────────────────────────────────────────

  □ EVERY modifier creation MUST be wrapped in try/except:
        try:
            mod = obj.modifiers.new("Bevel", 'BEVEL')
            mod.width = 0.0003
            mod.segments = 2
            mod.limit_method = 'ANGLE'
            mod.angle_limit = radians(30)
        except Exception:
            pass

  □ Modifier property assignments that may not exist in Blender 5.0 MUST be
    individually wrapped:
        try: bev.harden_normals = True
        except: pass
        try: sub.quality = 3
        except: pass
        try: sub.boundary_smooth = 'ALL'
        except: pass

  □ NEVER call bpy.ops.object.modifier_apply(). Leave modifiers on stack.

─── AUDIT 8: GEOMETRY INTEGRITY ─────────────────────────────────────────────

  □ Every function that creates geometry MUST produce at least 1 face.
    A function that returns an object with 0 faces is a silent failure.

  □ Smooth shading on metal parts MUST be set AFTER bm.to_mesh(), on the
    Mesh object's polygons (NOT on BMFace):
        for poly in mesh_data.polygons:
            poly.use_smooth = True

  □ Flat shading on gems — same rule:
        for poly in gem_mesh.polygons:
            poly.use_smooth = False

  □ recalc_face_normals MUST be called before bm.to_mesh():
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

  □ mesh.update() SHOULD be called after bm.to_mesh(mesh):
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

─── AUDIT 9: BAND CLOSURE — 360° RING MANDATORY ───────────────────────────

  □ The band/shank MUST sweep the FULL 360 degrees (0 to 2*PI).
    Search for the band sweep loop. Verify:
      - theta goes from 0 to 2*PI EXCLUSIVE: theta = 2*PI * i / n_major
      - Do NOT use range(n_major + 1) — this duplicates verts at 0 and 2*PI
      - Do NOT use i / (n_major - 1) — this also creates a gap

  □ CLOSING FACES between last and first profile loop are MANDATORY:
    After the sweep loop, there MUST be code that connects the last loop
    back to the first loop:
        for j in range(n_minor):
            j_next = (j + 1) % n_minor
            bm.faces.new([loops[n_major-1][j], loops[n_major-1][j_next],
                          loops[0][j_next], loops[0][j]])
    WITHOUT this, the band has a visible GAP = BROKEN RING.

  □ If the band is built as a torus and has an open seam/gap visible in
    the error screenshots, ADD the closing faces. This is the #1 most
    visible defect in any ring generation.

  □ BAND MUST BE A SOLID TORUS — NOT A HOLLOW SHELL OR RIBBON:
    The band cross-section profile MUST form a CLOSED LOOP — both the
    OUTER surface (visible from outside) AND the INNER surface (finger
    hole side) must have faces. If the band looks hollow or open when
    viewed from inside the ring hole, the inner wall is missing.
    The profile loop must go: outer top → outer side → inner side → inner
    top → back to start, forming a complete closed ring cross-section.
    A band with only an outer wall and no inner wall = BROKEN GEOMETRY.

═══════════════════════════════════════════════════════════════════════════════
PHASE 3: MENTAL EXECUTION — DRY RUN THE ENTIRE SCRIPT
═══════════════════════════════════════════════════════════════════════════════

Before outputting the fixed script, mentally execute it from line 1 to the end:
  1. Trace imports — all modules available?
  2. Trace utility function definitions — all defined before use?
  3. Trace build() — ring_col created? Passed to every helper?
  4. Trace every helper call — correct arguments? Return values used correctly?
  5. Trace every bmesh operation — ensure_lookup_table called? Faces wrapped?
  6. Trace every modifier — wrapped in try/except?
  7. Verify: ZERO NameErrors, ZERO TypeErrors, ZERO AttributeErrors,
     ZERO KeyErrors, ZERO SyntaxErrors, ZERO ValueErrors.

If you find ANY issue during this mental execution, FIX IT before outputting.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT RULES — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════════════════════

1. Output ONLY the complete fixed Python script. No explanations. No markdown.
2. Fix the reported error AND every hidden error found in the audit.
3. Preserve the ring geometry — only fix what is broken or dangerous.
4. Keep working code IDENTICAL — do not rewrite functions that work correctly.
5. The main function MUST be named build().
6. ring_col MUST be created inside build() and passed to all helpers.
7. NO materials, NO shaders, NO lighting, NO cameras, NO scene setup code.
8. ONLY bmesh for geometry. NO bpy.ops.mesh.* or bpy.ops.transform.*.
9. The script MUST run with: blender --background --python script.py
   with ZERO errors, ZERO warnings, producing valid geometry.
10. Return ONLY Python code. No explanations before or after. No markdown fences."""

    return base_prompt

# ========================= MODULE EXTRACTION (Smart Edit) =========================

def extract_module_code(code, func_name):
    """Extract a single function's source from the full script."""
    lines = code.split('\n')
    start = None
    indent = None
    for i, line in enumerate(lines):
        if re.match(rf'^(\s*)def\s+{re.escape(func_name)}\s*\(', line):
            start = i
            indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return None, None, None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == '':
            continue
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        if current_indent <= indent and stripped != '' and not stripped.startswith('#'):
            end = i
            break

    return '\n'.join(lines[start:end]), start, end


def splice_module_code(full_code, new_func_code, start, end):
    """Replace lines start:end with new function code."""
    lines = full_code.split('\n')
    new_lines = lines[:start] + new_func_code.split('\n') + lines[end:]
    return '\n'.join(new_lines)

# ========================= LLM API =========================

def _call_openrouter(or_model, system, prompt, max_tokens=200000, image_data=None, image_mime=None, in_cost=1.0, out_cost=10.0, label="LLM", extra_body=None):
    """Unified OpenRouter call for ALL models — same max_tokens, same retry, no timeout."""
    if not openrouter_client:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    log(f"Calling {label} via OpenRouter ({or_model}, max_tokens={max_tokens}, image={'yes' if image_data else 'no'})...", label)
    t0 = time.time()

    raw_prompt = f"{system}\n\n---\n\nUser Request: {prompt}"

    if image_data and image_mime:
        b64 = base64.b64encode(image_data).decode("utf-8")
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
            {"type": "text", "text": raw_prompt}
        ]
    else:
        user_content = [{"type": "text", "text": raw_prompt}]

    # Auto-retry on overload errors (NO LIMIT - keep trying forever)
    max_retries = 999
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": or_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_content}]
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = openrouter_client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""

            usage_info = {"model": or_model}
            if hasattr(response, 'usage') and response.usage:
                input_tok = response.usage.prompt_tokens or 0
                output_tok = response.usage.completion_tokens or 0
                usage_info = {
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "model": or_model,
                    "input_cost_per_mtok": in_cost,
                    "output_cost_per_mtok": out_cost,
                }
                usage_info["cost_usd"] = round(
                    input_tok / 1_000_000 * in_cost +
                    output_tok / 1_000_000 * out_cost, 4)
                log(f"{label} ({or_model}) tokens: in={input_tok}, out={output_tok}, cost=${usage_info['cost_usd']}", label)

            elapsed = time.time() - t0
            log(f"{label} responded: {elapsed:.1f}s, {len(raw)} chars", label)
            return _extract_code(raw), usage_info

        except Exception as e:
            err_str = str(e).lower()
            is_overloaded = (
                'overloaded' in err_str or
                '529' in err_str or
                '503' in err_str
            )
            if is_overloaded and attempt < max_retries:
                wait = attempt * 15  # 15s, 30s, 45s
                log(f"{label} overloaded (attempt {attempt}/{max_retries}), retrying in {wait}s...", label)
                time.sleep(wait)
                continue
            else:
                raise


def call_claude(system, prompt, max_tokens=60000, image_data=None, image_mime=None, model="claude-opus-4-6"):
    if "sonnet" in model:
        or_model = CLAUDE_SONNET_MODEL
        in_cost, out_cost = 3.0, 15.0
    else:
        or_model = CLAUDE_OPUS_MODEL
        in_cost, out_cost = 15.0, 75.0
    return _call_openrouter(or_model, system, prompt, max_tokens, image_data, image_mime, in_cost, out_cost, "CLAUDE")


def call_gemini(system, prompt, image_data=None, image_mime=None):
    return _call_openrouter(GEMINI_MODEL, system, prompt, 200000, image_data, image_mime, 1.25, 10.0, "GEMINI", extra_body={"reasoning": {"effort": "high"}})


def call_codex(system, prompt, image_data=None, image_mime=None):
    return _call_openrouter(CODEX_MODEL, system, prompt, 200000, image_data, image_mime, 1.75, 14.0, "CODEX")


def call_llm(llm_name, system, prompt, image_data=None, image_mime=None):
    """Returns (code, usage_info) tuple."""
    if llm_name == "gemini":
        return call_gemini(system, prompt, image_data=image_data, image_mime=image_mime)
    elif llm_name == "codex":
        return call_codex(system, prompt, image_data=image_data, image_mime=image_mime)
    else:
        # Route to correct Claude model
        if llm_name == "claude-sonnet":
            model = "claude-sonnet-4-6"
        else:  # "claude" or "claude-opus"
            model = "claude-opus-4-6"
        return call_claude(system, prompt, image_data=image_data, image_mime=image_mime, model=model)


def _compute_cost_summary(usage_list):
    """Aggregate multiple usage entries into a cost summary."""
    total_input = sum(u.get("input_tokens", 0) for u in usage_list)
    total_output = sum(u.get("output_tokens", 0) for u in usage_list)
    total_cost = sum(u.get("cost_usd", 0) for u in usage_list)
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_usd": round(total_cost, 4),
        "calls": len(usage_list),
        "details": usage_list
    }


def _extract_code(raw):
    """Pull Python code out of markdown fences or raw text."""
    if "```python" in raw:
        code = raw.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        code = raw.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        code = raw.strip()
    return code

# ========================= CODE SAFETY =========================

def preprocess_code(code):
    """Inject _safe_face helper and wrap bm.faces.new([...]) calls."""

    safe_helper = '''
# ===== AUTO-INJECTED SAFETY =====
def _safe_face(_bm_arg, _verts_arg):
    try:
        if len(_verts_arg) < 3:
            return None
        if len(set(id(v) for v in _verts_arg)) != len(_verts_arg):
            return None
        return _bm_arg.faces.new(_verts_arg)
    except (ValueError, IndexError, TypeError):
        return None
# ===== END SAFETY =====
'''
    # Remove any LLM-generated _safe_face definitions to avoid duplicates
    code = re.sub(
        r'(?m)^def _safe_face\(.*?\):\s*\n(?:[ \t]+.*\n)*',
        '# (duplicate _safe_face removed — using auto-injected version)\n',
        code
    )

    # Inject at the very TOP of the script (line 0) so it's defined before any usage
    code = safe_helper + '\n' + code

    # Replace bm.faces.new([...]) with _safe_face(bm, [...]) — single-line patterns
    code = re.sub(
        r'(\w+)\.faces\.new\((\[.*?\])\)',
        r'_safe_face(\1, \2)',
        code
    )

    return code

# ========================= BLENDER EXECUTION =========================

def run_blender(script_code, glb_output_path):
    session_dir = os.path.dirname(glb_output_path)
    script_path = os.path.join(session_dir, "ring_script.py")

    # Delete stale GLB from previous attempts so we don't falsely detect success
    if os.path.exists(glb_output_path):
        os.remove(glb_output_path)

    # Pre-process
    script_code = preprocess_code(script_code)

    # Remove if __name__ guard — we call build() ourselves
    script_code = re.sub(
        r'if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n\s*build\(\)',
        '# (build call moved to auto-export section)',
        script_code
    )

    # Prepend scene-clear: remove ALL default objects before anything runs
    scene_clear = """
# ========================= AUTO SCENE CLEAR =========================
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    bpy.data.collections.remove(c)
for m in list(bpy.data.meshes):
    bpy.data.meshes.remove(m)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)
print("[PIPELINE] Scene cleared")
# ========================= END SCENE CLEAR =========================

"""
    script_code = scene_clear + script_code

    export_code = f"""

# ========================= AUTO BUILD + EXPORT =========================
import bpy, os, traceback as _tb
from mathutils import Vector

_output = r"{glb_output_path}"
os.makedirs(os.path.dirname(_output), exist_ok=True)

print("[PIPELINE] Running build()...")
_build_error = False
try:
    build()
    print("[PIPELINE] build() completed")
except Exception as _be:
    print(f"[PIPELINE] build() error: {{_be}}")
    _tb.print_exc()
    _build_error = True
    print("[PIPELINE] BUILD FAILED — will NOT export partial geometry")

_obj_count = len([o for o in bpy.data.objects if o.type == 'MESH'])
print(f"[PIPELINE] Scene has {{_obj_count}} mesh objects")

# ========================= SPATIAL REPORT GENERATION =========================
print("===SPATIAL_REPORT_START===")
try:
    for _obj in bpy.data.objects:
        if _obj.type == 'MESH':
            _mesh = _obj.data
            _loc = _obj.location
            _rot = _obj.rotation_euler
            _scale = _obj.scale

            # Geometry stats
            _verts = len(_mesh.vertices)
            _edges = len(_mesh.edges)
            _faces = len(_mesh.polygons)

            # Bounding box in world space
            _bbox = [_obj.matrix_world @ Vector(v) for v in _obj.bound_box]
            _bbox_min = Vector((min([v.x for v in _bbox]), min([v.y for v in _bbox]), min([v.z for v in _bbox])))
            _bbox_max = Vector((max([v.x for v in _bbox]), max([v.y for v in _bbox]), max([v.z for v in _bbox])))

            # Parent info
            _parent = _obj.parent.name if _obj.parent else "None"

            # Modifiers
            _mods = [m.type for m in _obj.modifiers]

            print(f"MESH: {{_obj.name}}")
            print(f"  Location: {{_loc.x:.4f}}, {{_loc.y:.4f}}, {{_loc.z:.4f}}")
            print(f"  Rotation: {{_rot.x:.4f}}, {{_rot.y:.4f}}, {{_rot.z:.4f}}")
            print(f"  Scale: {{_scale.x:.4f}}, {{_scale.y:.4f}}, {{_scale.z:.4f}}")
            print(f"  Geometry: {{_verts}} verts, {{_edges}} edges, {{_faces}} faces")
            print(f"  BBox Min: {{_bbox_min.x:.4f}}, {{_bbox_min.y:.4f}}, {{_bbox_min.z:.4f}}")
            print(f"  BBox Max: {{_bbox_max.x:.4f}}, {{_bbox_max.y:.4f}}, {{_bbox_max.z:.4f}}")
            print(f"  Parent: {{_parent}}")
            print(f"  Modifiers: {{_mods}}")
            print("---")
except Exception as _spatial_err:
    print(f"Spatial report generation failed: {{_spatial_err}}")
print("===SPATIAL_REPORT_END===")
# ========================= END SPATIAL REPORT =========================

if _build_error:
    print("[PIPELINE] SKIPPING EXPORT — build() had errors, partial geometry is not valid")
elif _obj_count == 0:
    print("[PIPELINE] No mesh objects to export!")
else:
    bpy.ops.object.select_all(action='SELECT')
    print(f"[PIPELINE] Exporting GLB to: {{_output}}")
    try:
        bpy.ops.export_scene.gltf(
            filepath=_output,
            export_format='GLB',
            use_selection=True,
            export_apply=True,
            export_animations=False,
            export_cameras=False,
            export_lights=False
        )
        _size = os.path.getsize(_output)
        print(f"[PIPELINE] GLB exported: {{_size}} bytes")
    except Exception as _e:
        print(f"[PIPELINE] Export FAILED: {{_e}}")
        _tb.print_exc()
"""

    full_script = script_code + export_code

    log(f"Writing script: {script_path}", "BLENDER")
    with open(script_path, 'w') as f:
        f.write(full_script)

    cmd = [BLENDER_EXEC, "-b", "--python", script_path]
    log(f"Running Blender...", "BLENDER")
    t0 = time.time()

    blender_env = os.environ.copy()
    blender_lib_dir = os.path.join(os.path.dirname(BLENDER_EXEC), "lib")
    blender_env["LD_LIBRARY_PATH"] = blender_lib_dir + ":" + blender_env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=blender_env)  # NO TIMEOUT
        elapsed = time.time() - t0

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        pipeline_lines = [l for l in stdout.split('\n') if '[PIPELINE]' in l]
        error_lines = [l for l in (stdout + '\n' + stderr).split('\n')
                       if 'Error' in l or 'Traceback' in l or 'error' in l.lower()]

        for line in pipeline_lines:
            log(f"  {line.strip()}", "BLENDER")

        # Extract spatial report from stdout
        spatial_report = ""
        if "===SPATIAL_REPORT_START===" in stdout and "===SPATIAL_REPORT_END===" in stdout:
            start_idx = stdout.find("===SPATIAL_REPORT_START===")
            end_idx = stdout.find("===SPATIAL_REPORT_END===")
            if start_idx != -1 and end_idx != -1:
                spatial_report = stdout[start_idx + len("===SPATIAL_REPORT_START==="):end_idx].strip()

        glb_exists = os.path.exists(glb_output_path)
        glb_size = os.path.getsize(glb_output_path) if glb_exists else 0
        # GLB must be at least 1KB to have real geometry (172 bytes = empty)
        success = glb_exists and glb_size > 1024

        log(f"Blender done: {elapsed:.1f}s, exit={result.returncode}, glb={glb_size}B, ok={success}", "BLENDER")

        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "pipeline_log": pipeline_lines,
            "error_lines": error_lines,
            "glb_exists": glb_exists,
            "glb_size": glb_size,
            "elapsed": elapsed,
            "script_path": script_path,
            "spatial_report": spatial_report
        }

    except subprocess.TimeoutExpired:
        log("Blender TIMEOUT (300s)", "ERROR")
        return {"success": False, "error": "Blender timeout 300s", "stdout": "", "stderr": "",
                "pipeline_log": [], "error_lines": ["TimeoutExpired"], "glb_size": 0}
    except Exception as e:
        log(f"Blender EXCEPTION: {e}", "ERROR")
        return {"success": False, "error": str(e), "stdout": "", "stderr": "",
                "pipeline_log": [], "error_lines": [str(e)], "glb_size": 0}

# ========================= AUTO RETRY =========================

def run_with_retry(llm_name, initial_code, glb_path, session_id=None, max_retries=MAX_ERROR_RETRIES):
    """Run Blender on code. On error, send code+error to LLM to fix. Up to max_retries.
    Returns (code, result, retry_log, extra_usage_list)."""
    retry_log = []
    extra_usage = []
    code = initial_code
    last_spatial_report = None  # Track spatial data from previous attempts

    for attempt in range(1, max_retries + 1):
        log(f"[ATTEMPT {attempt}/{max_retries}]", "RETRY")
        if session_id:
            update_progress(session_id, "blender", f"Running Blender (attempt {attempt}/{max_retries})", attempt, max_retries)

        result = run_blender(code, glb_path)

        # Capture spatial report even if attempt fails (but geometry was generated)
        if result.get("spatial_report"):
            last_spatial_report = result["spatial_report"]

        entry = {
            "attempt": attempt,
            "success": result["success"],
            "code_length": len(code),
            "error_text": "",
            "timestamp": datetime.now().isoformat()
        }

        if result["success"]:
            retry_log.append(entry)
            log(f"[ATTEMPT {attempt}] SUCCESS", "RETRY")
            if session_id:
                update_progress(session_id, "done", f"Blender succeeded on attempt {attempt}")
            return code, result, retry_log, extra_usage

        # Extract error for retry
        error_text = '\n'.join(result.get("error_lines", [])[:20])
        stderr_tail = result.get("stderr", "")[-1500:]
        if stderr_tail:
            error_text += '\n' + stderr_tail
        entry["error_text"] = error_text[:3000]
        retry_log.append(entry)

        if attempt < max_retries:
            log(f"[ATTEMPT {attempt}] FAILED — asking LLM to fix...", "RETRY")
            if session_id:
                short_err = result.get("error_lines", ["Unknown error"])[:3]
                update_progress(session_id, "fixing",
                    f"Attempt {attempt} failed: {short_err[0][:80] if short_err else 'error'}. Enhancing design...",
                    attempt, max_retries)
            try:
                code, fix_usage = call_llm(llm_name, SYSTEM_PROMPT, build_fix_prompt(code, error_text[:2000], spatial_report=last_spatial_report))
                if fix_usage:
                    extra_usage.append(fix_usage)
                if session_id:
                    update_progress(session_id, "fixed", f"LLM returned fixed code ({len(code)} chars). Retrying...", attempt, max_retries)
            except Exception as e:
                log(f"LLM fix call failed: {e}", "ERROR")
                if session_id:
                    update_progress(session_id, "error", f"LLM fix call failed: {e}", attempt, max_retries)
                break
        else:
            log(f"[ATTEMPT {attempt}] FAILED — no more retries", "RETRY")
            if session_id:
                update_progress(session_id, "failed", f"All {max_retries} attempts failed", attempt, max_retries)

    return code, result, retry_log, extra_usage

# ========================= SESSION =========================

def get_session_dir(session_id):
    d = SESSIONS_DIR / session_id
    d.mkdir(exist_ok=True)
    return d

def save_session(session_id, data):
    with open(get_session_dir(session_id) / "session.json", 'w') as f:
        json.dump(data, f, indent=2)

def load_session(session_id):
    f = SESSIONS_DIR / session_id / "session.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None

def add_version_to_history(session, description, cost=0):
    """Add current code to version history"""
    if "version_history" not in session:
        session["version_history"] = []

    session["version_history"].append({
        "version": session["version"],
        "code": session["code"],
        "modules": session.get("modules", []),
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "cost": cost
    })
    session["current_version"] = session["version"]
    return session

# ========================= HELPERS =========================

def extract_modules(code):
    """Pull function names from code, excluding utility functions."""
    skip = {'nuke', 'build', 'mk', 'quad_bridge', 'make_circle_verts',
            'set_smooth', 'add_subsurf', 'add_bevel', 'add_solidify',
            'ngon', 'safe_set', '_safe_face'}
    modules = []
    for line in code.split('\n'):
        ls = line.strip()
        if ls.startswith('def ') and '(' in ls:
            fname = ls.split('def ')[1].split('(')[0]
            if fname not in skip:
                modules.append(fname)
    return modules

# ========================= VALIDATION FUNCTIONS (TWO-STAGE) =========================

def validate_with_model(screenshots_b64, code, user_prompt, master_prompt, model_name, reference_image_b64=None):
    """
    UNIFIED VALIDATION: Check both geometry integration AND aesthetics together.
    Includes reference sketch/image comparison for sketch-to-3D pipeline.
    """
    log(f"Validating ring with {model_name} ({len(screenshots_b64)} screenshots, ref_image={'yes' if reference_image_b64 else 'no'})...", "VALIDATION")

    ref_section = ""
    ref_priority = ""
    if reference_image_b64:
        ref_section = """
═══════════════════════════════════════════════════════════════════════
REFERENCE IMAGE COMPARISON — THIS IS THE #1 PRIORITY
═══════════════════════════════════════════════════════════════════════
THE VERY FIRST IMAGE is the USER'S ORIGINAL REFERENCE SKETCH/PHOTO — the GROUND TRUTH.
The remaining 20 images are rendered views of the 3D model that was generated from it.

Your PRIMARY job is to make the 3D model match the reference image EXACTLY.
Do a FORENSIC COMPARISON:
- Count every stone/flower/element in the reference → does the 3D have the SAME count?
- Check the EXACT shape of the band → does the 3D band match?
- Check the EXACT silhouette/outline → does the 3D match from the same angle?
- Check decorative elements (flowers, vines, filigree, openwork) → are they ALL present?
- Check proportions → is the 3D model the same proportions as the reference?
- If the reference shows a CABOCHON (smooth dome stone) → the 3D must NOT have a faceted diamond
- If the reference shows an OPEN/CUFF ring → the 3D must NOT be a closed band
- If the reference shows SPECIFIC petal shapes/counts → the 3D must match EXACTLY

ANY deviation from the reference image = a bug that MUST be fixed in the code.
"""
        ref_priority = """
PRIORITY 0 — MATCH THE REFERENCE IMAGE (HIGHEST PRIORITY):
Compare the 3D rendered views against the reference sketch/photo. Fix ANY deviation:
- Missing elements that appear in the reference → ADD them
- Extra elements NOT in the reference → REMOVE them
- Wrong shapes (e.g., faceted diamond where reference shows cabochon) → FIX shape
- Wrong proportions → ADJUST dimensions
- Wrong count of stones/flowers/elements → FIX count
- Wrong band type (closed vs open, thin vs wide) → FIX to match reference
- Band opening/gap is too wide → SHRINK it. Cuff/open bands must have a TINY gap (max 2-4% of circle, ~7-14 degrees). Wide openings look broken.
- Missing decorative details (filigree, openwork, texture) → ADD them
This is MORE important than any structural fix below.

"""

    validation_prompt = f"""USER'S ORIGINAL REQUEST:
{user_prompt}
{ref_section}
THE WORKING CODE THAT GENERATED THIS RING:
```python
{code}
```

{"The FIRST image is the reference sketch/photo. The NEXT" if reference_image_b64 else "I'm showing you"} 20 rendered views of this ring:

VIEWS 1-8: Full ring from 8 angles (front, back, left, right, top, bottom, and 2 angled views).
VIEWS 9-12: Smart framing shots — the ring fills the screen vertically showing both top and bottom, from front/back/left/right.
VIEWS 13-16: Zoomed-in close-ups of the TOP of the ring (diamond, prongs, setting area) from bird's-eye, front, left, and right.
VIEWS 17-20: Prong/setting detail shots — close-ups of the prongs and setting area from front, left, right, and back.

Pay extra attention to views 9-20 — the smart framing, top zoom, and prong detail shots reveal prong, diamond, setting, and structural problems most clearly.
{"MOST IMPORTANTLY: Compare EVERY rendered view against the reference sketch/photo. The 3D model must be a PIXEL-PERFECT match to the reference." if reference_image_b64 else ""}

You are given the original code that generated this ring. Look at every screenshot closely and fix every structural problem you find. Output the COMPLETE corrected code.

═══════════════════════════════════════════════════════════════════════
WHAT TO CHECK AND FIX — IN ORDER OF PRIORITY
═══════════════════════════════════════════════════════════════════════

{ref_priority}PRIORITY 1 — PRONGS MUST HOLD THE DIAMOND CORRECTLY:
Look closely at the prongs. This is the #1 problem in every generation.
- If prongs are floating in the air and NOT touching the band, fix them.
- If prongs are NOT gripping the diamond at its girdle, fix them.
- If prongs stab THROUGH the diamond instead of wrapping around it, pull them outward.
- You CAN completely rebuild the prongs from scratch if needed.

PRIORITY 2 — REMOVE ALL FLAT DISCS, PLATES, AND UFO SHAPES:
Look for any flat horizontal disc, plate, platform, or UFO-shaped object. REMOVE them.

PRIORITY 3 — BAND MUST BE A COMPLETE CLOSED 360 SOLID TORUS:
If the band has a GAP, SEAM, or OPENING — fix it. The band must sweep 360 degrees.
{"EXCEPTION: If the reference image shows an OPEN/CUFF ring, keep a gap — but make it TINY (max 2-4% of the full circle, ~7-14 degrees). Real cuffs have very narrow openings." if reference_image_b64 else ""}

PRIORITY 4 — EVERY DIAMOND MUST SIT IN A PROPER HOLDER/SEAT (ZERO FLOATING):
No diamond may EVER float in the air. Every diamond must have a visible metal holder.

PRIORITY 5 — REMOVE ANY BULKY/WEIRD TOP STRUCTURE:
If there is a massive bucket or cage around the diamond — REMOVE IT.
{"EXCEPTION: If the reference image shows a large/prominent setting, keep it!" if reference_image_b64 else ""}

PRIORITY 6 — REMOVE INCOMPLETE OR BROKEN MESHES:
Look for any mesh that looks half-built, corrupted, or nonsensical — REMOVE IT.

═══════════════════════════════════════════════════════════════════════
RULES FOR YOUR CORRECTIONS
═══════════════════════════════════════════════════════════════════════

1. Start with the EXISTING code as your base — do NOT rewrite from scratch.
2. Keep ALL function names and signatures the same.
3. You CAN delete code that creates flat discs, UFO shapes, broken meshes.
4. You CAN completely rewrite prong code if prongs are wrong.
5. You CAN adjust position values, dimensions, rotation, scale.
6. You CAN add entirely NEW geometry functions if the reference image has elements missing from the 3D model.
7. The code MUST still call build() and produce valid Blender 5.0 geometry.
8. Every distinct object MUST remain a separate mesh.
9. ALL faceted gems MUST use the locked build_brilliant_gem() function. For cabochons, build smooth domes.
10. Do NOT add any materials, lighting, cameras — geometry only.
11. Do NOT use bpy.ops for geometry creation — bmesh only.

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

First: In 2-3 sentences, describe what problems you found{"and how the 3D differs from the reference image" if reference_image_b64 else ""}.
Then: Return the COMPLETE corrected Python code inside ```python ... ``` fences.

IMPORTANT: There is ALWAYS something to improve. No ring is ever perfect on first attempt.
You MUST return corrected code. ALWAYS return COMPLETE corrected Python code."""

    try:
        # Prepare reference image (if provided) — goes FIRST
        ref_image_data = None
        if reference_image_b64:
            if ',' in reference_image_b64:
                mime_part, b64_data = reference_image_b64.split(',', 1)
                mime = mime_part.split(':')[1].split(';')[0]
                ref_image_data = {'mime': mime, 'data': b64_data}
            else:
                # Raw base64 without data URI prefix — assume JPEG
                ref_image_data = {'mime': 'image/jpeg', 'data': reference_image_b64}

        # Prepare screenshot data for LLM
        images_data = []
        for b64_uri in screenshots_b64:
            if ',' in b64_uri:
                mime_part, b64_data = b64_uri.split(',', 1)
                mime = mime_part.split(':')[1].split(';')[0]
                images_data.append({'mime': mime, 'data': b64_data})

        # Call the same LLM that generated the ring
        if model_name in ("gemini", "gemini-3-pro-preview", "codex"):
            or_model = GEMINI_MODEL if model_name != "codex" else CODEX_MODEL
            content = []
            # Reference image FIRST
            if ref_image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{ref_image_data['mime']};base64,{ref_image_data['data']}"}
                })
            for img in images_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime']};base64,{img['data']}"}
                })
            content.append({
                "type": "text",
                "text": f"You are a luxury jewelry design critic.\n\n{validation_prompt}"
            })
            extra = {"reasoning": {"effort": "high"}} if model_name != "codex" else {}
            response = openrouter_client.chat.completions.create(
                model=or_model,
                max_tokens=200000,
                messages=[{"role": "user", "content": content}],
                extra_body=extra
            )
            response_text = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

        else:
            content = []
            # Reference image FIRST
            if ref_image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{ref_image_data['mime']};base64,{ref_image_data['data']}"}
                })
            for img in images_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime']};base64,{img['data']}"}
                })
            content.append({
                "type": "text",
                "text": f"You are a luxury jewelry design critic.\n\n{validation_prompt}"
            })

            if "sonnet" in model_name.lower():
                validation_model = CLAUDE_SONNET_MODEL
            else:
                validation_model = CLAUDE_OPUS_MODEL

            log(f"Validating with Claude model: {validation_model} (generation was with {model_name})", "VALIDATION")

            response = openrouter_client.chat.completions.create(
                model=validation_model,
                max_tokens=200000,
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

        # Calculate cost
        if model_name in ("gemini", "gemini-3-pro-preview"):
            input_cost_per_mtok = 2.50
            output_cost_per_mtok = 10.0
        elif "opus" in model_name.lower():
            input_cost_per_mtok = 5.0
            output_cost_per_mtok = 25.0
        else:  # sonnet
            input_cost_per_mtok = 3.0
            output_cost_per_mtok = 15.0

        cost = (tokens_in / 1_000_000) * input_cost_per_mtok + (tokens_out / 1_000_000) * output_cost_per_mtok

        log(f"Validation tokens: in={tokens_in}, out={tokens_out}, cost=${cost:.4f}", "VALIDATION")
        log(f"Validation response: {len(response_text)} chars, starts: {response_text[:200]!r}", "VALIDATION")

        # --- Extract code FIRST ---
        corrected_code = None

        # Strategy 1: Standard ```python ... ``` fences
        code_match = re.search(r'```python\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
        if code_match:
            corrected_code = code_match.group(1).strip()
            log(f"Code extracted (fenced): {len(corrected_code)} chars", "VALIDATION")

        # Strategy 2: Opening ```python but output truncated
        if not corrected_code and '```python' in response_text:
            after_fence = response_text.split('```python', 1)[1]
            if '```' in after_fence:
                after_fence = after_fence.split('```')[0]
            after_fence = after_fence.strip()
            if len(after_fence) > 200 and ('def ' in after_fence or 'import ' in after_fence):
                corrected_code = after_fence
                log(f"Code extracted (truncated fence fallback): {len(corrected_code)} chars", "VALIDATION")

        # Strategy 3: No fences — raw code block starting with import
        if not corrected_code:
            raw_match = re.search(r'(import bpy\b.+)', response_text, re.DOTALL)
            if raw_match:
                raw_code = raw_match.group(1).strip()
                if len(raw_code) > 500 and 'def build' in raw_code:
                    corrected_code = raw_code
                    log(f"Code extracted (raw import fallback): {len(corrected_code)} chars", "VALIDATION")

        if corrected_code:
            log(f"Validation result: INVALID — returning {len(corrected_code)} chars corrected code", "VALIDATION")
            return {
                "is_valid": False,
                "message": "Generating more beautiful design...",
                "corrected_code": corrected_code,
                "cost": cost,
                "tokens": {"input": tokens_in, "output": tokens_out},
                "full_response": response_text
            }

        log(f"WARNING: Could not extract code from validation response.", "VALIDATION")
        return {
            "is_valid": False,
            "message": "Validation failed to return corrected code — retrying",
            "cost": cost,
            "tokens": {"input": tokens_in, "output": tokens_out}
        }

    except Exception as e:
        log(f"Validation EXCEPTION (not silencing): {e}", "ERROR")
        traceback.print_exc()
        return {
            "is_valid": False,
            "message": f"Validation error: {str(e)[:100]}",
            "cost": 0,
            "tokens": {"input": 0, "output": 0}
        }


# ========================= ROUTES =========================

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(PROJECT_ROOT / "static"), filename)


@app.route("/")
def index():
    return send_file(str(PROJECT_ROOT / "index.html"))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Kick off generation from an uploaded sketch/image. Returns session_id immediately.
    Frontend polls /api/progress/<session_id> until stage is 'done' or 'failed'.
    IMAGE IS REQUIRED. Text description is optional context."""
    image_data = None
    image_mime = None
    prompt = ""

    if request.content_type and "multipart" in request.content_type:
        prompt = request.form.get("prompt", "").strip()
        llm_name = request.form.get("llm", "gemini").strip().lower()
        if "image" in request.files:
            img_file = request.files["image"]
            if img_file.filename:
                image_data = img_file.read()
                image_mime = img_file.content_type or "image/jpeg"
                log(f"Received sketch image: {img_file.filename} ({len(image_data)} bytes, {image_mime})", "API")
    else:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        llm_name = data.get("llm", "gemini").strip().lower()
        # Support base64 image in JSON body
        if data.get("image_base64"):
            image_data = base64.b64decode(data["image_base64"])
            image_mime = data.get("image_mime", "image/jpeg")

    if not image_data:
        return jsonify({"error": "Please upload a sketch or reference image of a ring"}), 400
    if llm_name not in ("claude", "claude-sonnet", "claude-opus", "gemini"):
        return jsonify({"error": "Invalid LLM. Use 'claude', 'claude-sonnet', 'claude-opus', or 'gemini'"}), 400

    session_id = f"s_{uuid.uuid4().hex[:10]}_{int(time.time())}"
    session_dir = get_session_dir(session_id)

    # Save reference image to session dir
    ext = image_mime.split("/")[-1] if image_mime else "jpg"
    img_path = session_dir / f"reference.{ext}"
    with open(img_path, "wb") as f:
        f.write(image_data)
    log(f"Saved reference sketch: {img_path}", "API")

    log(f"=== SKETCH-TO-3D [{llm_name.upper()}]: '{prompt[:60] if prompt else '[IMAGE-ONLY]'}' ===", "API")
    update_progress(session_id, "llm", f"Analyzing sketch with {llm_name.upper()}...")

    # Run the full pipeline in a background thread
    def _run_pipeline():
        glb_path = str(session_dir / "model.glb")

        # Step 1: Call LLM with the sketch image
        try:
            gen_prompt = build_sketch_generation_prompt(prompt)
            initial_code, usage_info = call_llm(llm_name, SYSTEM_PROMPT, gen_prompt, image_data=image_data, image_mime=image_mime)
        except Exception as e:
            log(f"[STEP 1] FAILED: {e}", "ERROR")
            err_lower = str(e).lower() + repr(e).lower()
            is_overloaded = 'overloaded' in err_lower or '529' in err_lower or getattr(e, 'status_code', None) == 529
            error_msg = "AI model is currently overloaded. Please try again in a moment." if is_overloaded else f"LLM API failed: {e}"
            update_progress(session_id, "failed", error_msg)
            return

        total_usage = [usage_info] if usage_info else []
        modules = extract_modules(initial_code)
        log(f"[STEP 1] Done. {len(initial_code)} chars, {initial_code.count(chr(10))} lines, modules: {modules}", "PIPELINE")

        # Step 2: Run Blender with auto-retry
        log("[STEP 2] Running Blender (with auto-retry)...", "PIPELINE")
        code, result, retry_log, retry_usage = run_with_retry(llm_name, initial_code, glb_path, session_id=session_id)
        total_usage.extend(retry_usage)

        cost_summary = _compute_cost_summary(total_usage)
        modules = extract_modules(code)
        skip_validation = False

        session_data = {
            "session_id": session_id,
            "prompt": prompt,
            "has_reference_image": True,
            "llm_name": llm_name,
            "code": code,
            "modules": modules,
            "version": 1,
            "current_version": 1,
            "edits": [],
            "version_history": [{
                "version": 1,
                "code": code,
                "modules": modules,
                "timestamp": datetime.now().isoformat(),
                "description": "Initial sketch-to-3D generation",
                "cost": cost_summary.get("total_usd", 0)
            }],
            "created": datetime.now().isoformat(),
            "retry_log": retry_log,
            "cost": cost_summary.get("total_usd", 0),
            "spatial_report": result.get("spatial_report", ""),
            "skip_validation": skip_validation,
            "blender_result": {
                "success": result["success"],
                "returncode": result.get("returncode"),
                "glb_size": result.get("glb_size", 0),
                "elapsed": result.get("elapsed", 0),
                "pipeline_log": result.get("pipeline_log", []),
                "error_lines": result.get("error_lines", [])
            }
        }
        save_session(session_id, session_data)

        if not result["success"]:
            log("[STEP 2] FAILED after all retries", "ERROR")
            error_lines = result.get("error_lines", [])
            short_err = error_lines[0][:120] if error_lines else "Blender failed after retries"
            update_progress(session_id, "failed", short_err)
            return

        log(f"=== GENERATE COMPLETE: {session_id} | cost=${cost_summary.get('total_usd',0):.4f} ===", "SUCCESS")
        update_progress(session_id, "done", "Design ready for client-side validation")

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return jsonify({
        "session_id": session_id,
        "success": True,
        "async": True
    })


@app.route("/api/edit", methods=["POST"])
def api_edit():
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    edit_request = data.get("edit_request", "").strip()
    llm_name = data.get("llm", "gemini").strip().lower()
    target_module = data.get("target_module", None)

    if not session_id or not edit_request:
        return jsonify({"error": "Missing session_id or edit_request"}), 400

    session = load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    log(f"=== EDIT [{llm_name.upper()}]: '{edit_request[:60]}' on {session_id} ===", "API")

    current_code = session["code"]
    session_dir = get_session_dir(session_id)
    glb_path = str(session_dir / "model.glb")

    # Step 1: Call LLM
    log("[STEP 1] Calling LLM for edit...", "PIPELINE")
    edit_usage = []
    try:
        if target_module:
            new_code, u = call_llm(llm_name, SYSTEM_PROMPT,
                                   build_smart_edit_prompt(current_code, edit_request, target_module))
            if u: edit_usage.append(u)
        else:
            new_code, u = call_llm(llm_name, SYSTEM_PROMPT,
                                build_edit_prompt(current_code, edit_request))
            if u: edit_usage.append(u)
    except Exception as e:
        log(f"[STEP 1] FAILED: {e}", "ERROR")
        err_lower = str(e).lower() + repr(e).lower()
        is_overloaded = 'overloaded' in err_lower or '529' in err_lower or getattr(e, 'status_code', None) == 529
        error_msg = "AI model is currently overloaded. Please try again in a moment." if is_overloaded else f"LLM API failed: {e}"
        return jsonify({"error": error_msg, "overloaded": is_overloaded}), 529 if is_overloaded else 500

    # Step 2: Run with retry
    code, result, retry_log, retry_usage = run_with_retry(llm_name, new_code, glb_path)
    edit_usage.extend(retry_usage)
    cost_summary = _compute_cost_summary(edit_usage)

    modules = extract_modules(code)

    session["code"] = code
    session["modules"] = modules
    session["version"] = session.get("version", 1) + 1
    session["edits"].append({
        "request": edit_request,
        "target_module": target_module,
        "llm": llm_name,
        "timestamp": datetime.now().isoformat(),
        "version": session["version"]
    })
    session["retry_log"] = retry_log
    session["blender_result"] = {
        "success": result["success"],
        "glb_size": result.get("glb_size", 0),
        "elapsed": result.get("elapsed", 0),
        "pipeline_log": result.get("pipeline_log", []),
        "error_lines": result.get("error_lines", [])
    }
    save_session(session_id, session)

    if not result["success"]:
        return jsonify({
            "error": "Blender failed after retries",
            "session_id": session_id,
            "code": code,
            "modules": modules,
            "retry_log": retry_log,
            "cost": cost_summary,
            "llm_used": llm_name,
            "blender_stdout": result.get("stdout", "")[-3000:],
            "blender_stderr": result.get("stderr", "")[-1000:],
            "pipeline_log": result.get("pipeline_log", []),
            "error_lines": result.get("error_lines", [])
        }), 500

    return jsonify({
        "session_id": session_id,
        "glb_url": f"/api/model/{session_id}",
        "code": code,
        "modules": modules,
        "version": session["version"],
        "glb_size": result["glb_size"],
        "retry_log": retry_log,
        "cost": cost_summary,
        "llm_used": llm_name
    })


@app.route("/api/regen-part", methods=["POST"])
def api_regen_part():
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    part_type = data.get("part_type", "").strip()
    description = data.get("description", "").strip()
    llm_name = data.get("llm", "gemini").strip().lower()

    if not session_id or not part_type:
        return jsonify({"error": "Missing session_id or part_type"}), 400

    session = load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    current_code = session["code"]
    session_dir = get_session_dir(session_id)
    glb_path = str(session_dir / "model.glb")

    user_desc = description if description else f"Regenerate the {part_type} with better aesthetics and integration"
    regen_prompt = build_part_regen_prompt(current_code, part_type, user_desc)

    regen_usage = []
    try:
        new_code, u = call_llm(llm_name, SYSTEM_PROMPT, regen_prompt)
        if u: regen_usage.append(u)
    except Exception as e:
        err_lower = str(e).lower() + repr(e).lower()
        is_overloaded = 'overloaded' in err_lower or '529' in err_lower or getattr(e, 'status_code', None) == 529
        error_msg = "AI model is currently overloaded. Please try again in a moment." if is_overloaded else f"LLM API failed: {e}"
        return jsonify({"error": error_msg, "overloaded": is_overloaded}), 529 if is_overloaded else 500

    code, result, retry_log, retry_usage = run_with_retry(llm_name, new_code, glb_path)
    regen_usage.extend(retry_usage)
    cost_summary = _compute_cost_summary(regen_usage)

    modules = extract_modules(code)

    session["code"] = code
    session["modules"] = modules
    session["version"] = session.get("version", 1) + 1
    session["edits"].append({
        "request": f"[REGEN {part_type}] {user_desc}",
        "target_module": part_type,
        "llm": llm_name,
        "timestamp": datetime.now().isoformat(),
        "version": session["version"]
    })
    session["retry_log"] = retry_log
    session["blender_result"] = {
        "success": result["success"],
        "glb_size": result.get("glb_size", 0),
        "elapsed": result.get("elapsed", 0),
        "pipeline_log": result.get("pipeline_log", []),
        "error_lines": result.get("error_lines", [])
    }
    save_session(session_id, session)

    if not result["success"]:
        return jsonify({
            "error": "Blender failed after retries",
            "session_id": session_id,
            "code": code,
            "modules": modules,
            "retry_log": retry_log,
            "cost": cost_summary,
            "llm_used": llm_name,
            "blender_stderr": result.get("stderr", "")[-1000:],
            "error_lines": result.get("error_lines", [])
        }), 500

    return jsonify({
        "session_id": session_id,
        "glb_url": f"/api/model/{session_id}",
        "code": code,
        "modules": modules,
        "version": session["version"],
        "glb_size": result["glb_size"],
        "retry_log": retry_log,
        "cost": cost_summary,
        "llm_used": llm_name
    })


@app.route("/api/generate-new-part", methods=["POST"])
def api_generate_new_part():
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    description = data.get("description", "").strip()
    llm_name = data.get("llm", "gemini").strip().lower()

    if not session_id or not description:
        return jsonify({"error": "Missing session_id or description"}), 400

    session = load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    current_code = session["code"]
    session_dir = get_session_dir(session_id)
    glb_path = str(session_dir / "model.glb")

    add_prompt = f"""Here is the COMPLETE existing ring script:

```python
{current_code}
```

The user wants to ADD A NEW PART to this ring:
"{description}"

INSTRUCTIONS:
1. Return the COMPLETE script with a NEW function added for this part.
2. ALL EXISTING functions must remain BYTE-FOR-BYTE IDENTICAL — do NOT touch them.
3. Add a new function (e.g. create_new_part()) that creates the requested geometry.
4. Call your new function from main() BEFORE the final export/join step.
5. The new part must integrate spatially with the existing ring dimensions (use the same shared variables).
6. ONLY bmesh geometry. NO materials, NO lighting.
7. Return ONLY Python code. No explanations. No markdown fences."""

    part_usage = []
    try:
        new_code, u = call_llm(llm_name, SYSTEM_PROMPT, add_prompt)
        if u: part_usage.append(u)
    except Exception as e:
        err_lower = str(e).lower() + repr(e).lower()
        is_overloaded = 'overloaded' in err_lower or '529' in err_lower or getattr(e, 'status_code', None) == 529
        error_msg = "AI model is currently overloaded. Please try again in a moment." if is_overloaded else f"LLM API failed: {e}"
        return jsonify({"error": error_msg, "overloaded": is_overloaded}), 529 if is_overloaded else 500

    code, result, retry_log, retry_usage = run_with_retry(llm_name, new_code, glb_path)
    part_usage.extend(retry_usage)
    cost_summary = _compute_cost_summary(part_usage)

    modules = extract_modules(code)

    session["code"] = code
    session["modules"] = modules
    session["version"] = session.get("version", 1) + 1
    session["edits"].append({
        "request": f"[NEW PART] {description}",
        "target_module": "new_part",
        "llm": llm_name,
        "timestamp": datetime.now().isoformat(),
        "version": session["version"]
    })
    session["retry_log"] = retry_log
    session["blender_result"] = {
        "success": result["success"],
        "glb_size": result.get("glb_size", 0),
        "elapsed": result.get("elapsed", 0),
        "pipeline_log": result.get("pipeline_log", []),
        "error_lines": result.get("error_lines", [])
    }
    save_session(session_id, session)

    if not result["success"]:
        return jsonify({
            "error": "Blender failed after retries",
            "session_id": session_id,
            "code": code,
            "modules": modules,
            "retry_log": retry_log,
            "cost": cost_summary,
            "llm_used": llm_name,
            "blender_stderr": result.get("stderr", "")[-1000:],
            "error_lines": result.get("error_lines", [])
        }), 500

    return jsonify({
        "session_id": session_id,
        "glb_url": f"/api/model/{session_id}",
        "code": code,
        "modules": modules,
        "version": session["version"],
        "glb_size": result["glb_size"],
        "retry_log": retry_log,
        "cost": cost_summary,
        "llm_used": llm_name
    })


@app.route("/api/model/<session_id>")
def api_model(session_id):
    glb = SESSIONS_DIR / session_id / "model.glb"
    if not glb.exists():
        return jsonify({"error": "Model not found"}), 404
    return send_file(str(glb), mimetype='model/gltf-binary')


@app.route("/api/download/<session_id>")
def api_download(session_id):
    glb = SESSIONS_DIR / session_id / "model.glb"
    if not glb.exists():
        return jsonify({"error": "Model not found"}), 404
    return send_file(str(glb), mimetype='model/gltf-binary',
                     as_attachment=True, download_name=f"ring_{session_id}.glb")


@app.route("/api/session/<session_id>")
def api_session(session_id):
    s = load_session(session_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


@app.route("/api/magic-texture", methods=["POST"])
def api_magic_texture():
    data = request.json or {}
    mesh_names = data.get("mesh_names", [])
    if not mesh_names:
        return jsonify({"error": "No mesh names provided"}), 400

    assignments = []
    gem_keywords = ['gem', 'diamond', 'stone', 'brilliant', 'ruby', 'sapphire',
                    'emerald', 'amethyst', 'topaz', 'culet', 'facet', 'crystal',
                    'jewel', 'accent_gem', 'halo_gem', 'center_gem', 'pave']
    prong_keywords = ['prong', 'claw', 'bead', 'milgrain']

    for name in mesh_names:
        lower = name.lower().replace(' ', '_')
        if any(kw in lower for kw in gem_keywords):
            assignments.append({"name": name, "material": "diamond"})
        elif any(kw in lower for kw in prong_keywords):
            assignments.append({"name": name, "material": "platinum"})
        else:
            assignments.append({"name": name, "material": "gold"})

    return jsonify({"assignments": assignments})


@app.route("/api/progress/<session_id>")
def api_progress(session_id):
    p = pipeline_progress.get(session_id)
    if not p:
        return jsonify({"stage": "unknown", "detail": "No progress data"})
    resp = dict(p)
    if resp.get("stage") == "done":
        resp["glb_url"] = f"/api/model/{session_id}"
        resp["needs_validation"] = True
    return jsonify(resp)


@app.route("/api/validate", methods=["POST"])
def api_validate():
    data = request.json or {}
    images = data.get("images", [])
    session_id = data.get("session_id", "")

    if not images or len(images) < 1:
        return jsonify({"valid": True, "message": "No images to validate"}), 200

    analysis_prompt = """You are an expert 3D jewelry model inspector. You are looking at a computer-generated 3D ring model from 6 angles.

Check for GEOMETRY DEFECTS (in priority order):
1. FLOATING/DISCONNECTED PARTS
2. PRONG-GEM INTERSECTION
3. DISASSEMBLED LOOK
4. BAND INCOMPLETE
5. MISSING GEOMETRY
6. PROPORTIONAL ERRORS

If the ring has no structural defects, respond with EXACTLY: VALID
If there IS a defect, respond with EXACTLY: ERROR: [one sentence describing the defect]"""

    try:
        content = []
        for img_uri in images:
            content.append({"type": "image_url", "image_url": {"url": img_uri}})
        content.append({"type": "text", "text": analysis_prompt})

        response = openrouter_client.chat.completions.create(
            model=GPT_VISION_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": content}]
        )
        result_text = (response.choices[0].message.content or "").strip()

        if result_text.upper().startswith("VALID"):
            return jsonify({"valid": True, "message": "Ring looks good!"})
        else:
            error_msg = result_text.replace("ERROR:", "").strip()
            return jsonify({"valid": False, "message": error_msg})

    except Exception as e:
        return jsonify({"valid": True, "message": f"Validation skipped: {str(e)[:100]}"}), 200


@app.route("/api/validate-with-screenshots", methods=["POST"])
def api_validate_with_screenshots():
    data = request.json or {}
    screenshots_b64 = data.get("screenshots", [])
    session_id = data.get("session_id", "")

    if not screenshots_b64:
        return jsonify({"is_valid": True, "message": "No screenshots provided"}), 200

    session_path = SESSIONS_DIR / session_id / "session.json"
    if not session_path.exists():
        return jsonify({"error": "Session not found"}), 404

    with open(session_path, 'r') as f:
        session = json.load(f)

    code = session.get("code", "")
    user_prompt = session.get("prompt", "")
    llm_name = session.get("llm_name", "gemini")

    model_name = "gemini-3-pro-preview"

    if not code:
        return jsonify({"error": "No code found in session"}), 400

    # Load reference image if it exists (could be reference.png, .jpg, .jpeg, .webp)
    reference_image_b64 = None
    session_dir = SESSIONS_DIR / session_id
    for ext in ["png", "jpg", "jpeg", "webp"]:
        ref_img_path = session_dir / f"reference.{ext}"
        if ref_img_path.exists():
            with open(ref_img_path, "rb") as rf:
                ref_bytes = rf.read()
                mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
                mime = mime_map.get(ext, "image/jpeg")
                reference_image_b64 = f"data:{mime};base64,{base64.b64encode(ref_bytes).decode('utf-8')}"
                log(f"Loaded reference image for validation: {ref_img_path} ({len(ref_bytes)} bytes)", "VALIDATION")
            break

    result = validate_with_model(screenshots_b64, code, user_prompt, SYSTEM_PROMPT, model_name, reference_image_b64=reference_image_b64)

    session["validation"] = {
        "screenshots": screenshots_b64,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

    if not result["is_valid"] and "corrected_code" in result:
        glb_path = str(SESSIONS_DIR / session_id / "model.glb")
        final_code, blender_result, val_retry_log, val_extra_usage = run_with_retry(
            llm_name, result["corrected_code"], glb_path, max_retries=3
        )

        if blender_result["success"]:
            session["code"] = final_code
            session["version"] = session.get("version", 1) + 1
            val_cost = result["cost"] + sum(u.get("cost_usd", 0) for u in val_extra_usage)
            session["cost"] = session.get("cost", 0) + val_cost
            session["spatial_report"] = blender_result.get("spatial_report", "")

            with open(session_path, 'w') as f:
                json.dump(session, f)

            return jsonify({
                "is_valid": result["is_valid"],
                "message": result["message"],
                "regenerated": True,
                "cost": val_cost
            })
        else:
            return jsonify({
                "is_valid": True,
                "message": "Validation corrections failed after retries, using original design",
                "regenerated": False,
                "cost": result["cost"]
            })

    session["cost"] = session.get("cost", 0) + result["cost"]
    with open(session_path, 'w') as f:
        json.dump(session, f)

    return jsonify({
        "is_valid": result["is_valid"],
        "message": result["message"],
        "regenerated": False,
        "cost": result["cost"]
    })


@app.route("/api/debug/<session_id>")
def api_debug(session_id):
    session_path = SESSIONS_DIR / session_id / "session.json"
    if not session_path.exists():
        return jsonify({"error": "Session not found"}), 404

    with open(session_path, 'r') as f:
        session = json.load(f)

    return jsonify({
        "prompt": session.get("prompt", ""),
        "code": session.get("code", ""),
        "modules": session.get("modules", []),
        "cost": session.get("cost", 0),
        "llm_name": session.get("llm_name", ""),
        "validation": session.get("validation", {}),
        "spatial_report": session.get("spatial_report", ""),
        "version": session.get("version", 1),
        "created": session.get("created", ""),
        "edits": session.get("edits", [])
    })


@app.route("/api/undo/<session_id>", methods=["POST"])
def api_undo(session_id):
    session = load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    current_ver = session.get("current_version", session.get("version", 1))

    if current_ver <= 1:
        return jsonify({"error": "Already at first version"}), 400

    target_version = current_ver - 1
    version_history = session.get("version_history", [])

    target_data = None
    for vh in version_history:
        if vh["version"] == target_version:
            target_data = vh
            break

    if not target_data:
        return jsonify({"error": "Version not found in history"}), 404

    session["code"] = target_data["code"]
    session["modules"] = target_data["modules"]
    session["current_version"] = target_version

    glb_path = str(SESSIONS_DIR / session_id / "model.glb")
    result = run_blender(target_data["code"], glb_path)

    if not result["success"]:
        return jsonify({"error": "Failed to regenerate previous version"}), 500

    save_session(session_id, session)

    return jsonify({
        "success": True,
        "version": target_version,
        "glb_url": f"/api/model/{session_id}",
        "description": target_data.get("description", f"Version {target_version}")
    })


@app.route("/api/reset/<session_id>", methods=["POST"])
def api_reset(session_id):
    session = load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    version_history = session.get("version_history", [])

    if not version_history:
        return jsonify({"error": "No version history"}), 404

    original = version_history[0]

    session["code"] = original["code"]
    session["modules"] = original["modules"]
    session["current_version"] = 1

    glb_path = str(SESSIONS_DIR / session_id / "model.glb")
    result = run_blender(original["code"], glb_path)

    if not result["success"]:
        return jsonify({"error": "Failed to regenerate original version"}), 500

    save_session(session_id, session)

    return jsonify({
        "success": True,
        "version": 1,
        "glb_url": f"/api/model/{session_id}",
        "description": original.get("description", "Original design")
    })


@app.route("/api/upload-part", methods=["POST"])
def api_upload_part():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".glb"):
        return jsonify({"error": "Only .glb files are supported"}), 400
    upload_id = str(uuid.uuid4())[:8]
    upload_dir = PROJECT_ROOT / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{upload_id}_{f.filename}"
    f.save(str(save_path))
    return jsonify({"success": True, "url": f"/static/uploads/{save_path.name}"})


@app.route("/api/reference-image/<session_id>")
def api_reference_image(session_id):
    """Serve the reference sketch image for a session."""
    session_dir = SESSIONS_DIR / session_id
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
        img_path = session_dir / f"reference.{ext}"
        if img_path.exists():
            return send_file(str(img_path))
    return jsonify({"error": "Reference image not found"}), 404


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "blender": os.path.exists(BLENDER_EXEC),
        "claude": bool(OPENROUTER_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY)
    })

# ========================= MAIN =========================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SKETCH-TO-3D JEWELRY PIPELINE")
    print("  Image-to-CAD | Claude + OpenRouter | Auto-Retry")
    print("=" * 60)
    print(f"  Blender: {BLENDER_EXEC} ({'OK' if os.path.exists(BLENDER_EXEC) else 'MISSING'})")
    print(f"  OpenRouter: {'OK' if OPENROUTER_API_KEY else 'NOT SET'} (all models)")
    print(f"  Claude Opus: {CLAUDE_OPUS_MODEL}")
    print(f"  Claude Sonnet: {CLAUDE_SONNET_MODEL}")
    print(f"  Gemini: {GEMINI_MODEL}")
    print(f"  Sessions: {SESSIONS_DIR}")
    print("=" * 60 + "\n")

    port = int(os.environ.get("PORT", 5003))
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
