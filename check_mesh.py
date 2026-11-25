"""Check that a 3D model on EmpirioLabs AI is the public model it is said to be, without running any model.

A fingerprint of kind "mesh" (fingerprints/<model>.json) was made once, from the model's public weights: for a fixed
set of probes (a picture of an object on a transparent background, kept as PNG in the folder beside the file, and a
seed), a coarse signature of the GLB the public pipeline makes. The mesh's bounding box is scaled by its longest side,
so a flat object stays flat, and three things are read inside it:

    shape    the share of the mesh's surface area in each cell of a 16 x 16 x 16 grid
    box      the box's proportions: each side over the longest
    colors   the average base color of the surface in each cell of an 8 x 8 x 8 grid, when the mesh has a PNG texture

Remeshing, decimation and another texture layout move vertices and faces and change their number, but hardly move any
surface from one cell to another, so the signature survives them. Another model, or the same model with another seed,
shapes the object's unseen sides, depth and details differently. The seed's noise is drawn as float32 numbers from
torch's CPU generator, so the same request gives the same mesh wherever the public weights run, up to the small
rounding differences between graphics cards and between runs of the mesh clean-up.

    python check_model.py --model trellis-2-4b                    the usual way, as for every other model
    python check_mesh.py check --model trellis-2-4b               the same check, --out keeps the numbers read
    python check_mesh.py signature model.glb                      the signature of a GLB you have
    python check_mesh.py compare model.glb other.glb              how far two GLBs are
    python check_mesh.py make --weights <folder> --name microsoft/TRELLIS.2-4B --revision <commit> \
        --model trellis-2-4b --probes <probes.json> --request '<API settings as JSON>' --pipeline trellis2 \
        --out fingerprints/trellis-2-4b.json --shape-distance <limit> --typical-distance <limit> --box-gap <limit> \
        --close <share> [--color-distance <limit>]

The distances, each with the limit the file states: shape, one minus the correlation of the two grids of surface
shares (0 the same shape, about 1 nothing in common); box, the largest difference between the two boxes' proportions;
colors, one minus the correlation of the colors of the cells both meshes have surface in.

The check uses only the Python standard library: it reads the GLB's JSON and binary chunks, its buffers, buffer views
and accessors (sparse and normalized ones included), every mesh of the scene with its node transforms, and a PNG base
color texture (check_image's PNG reader). A mesh compressed with Draco or meshopt is not read. It reads your API key
from EMPIRIOLABS_API_KEY and asks for each probe's mesh once, billed to your account at the model's listed price (the
smallest settings the model takes). A probe's picture is sent from the folder beside the fingerprint, and only when its
SHA-256 is the one the fingerprint records. `make` runs the public weights (--pipeline trellis2: the TRELLIS.2 code
published with them, and its own requirements) or reads GLBs another public program made, one per probe named
<seed>.glb (--meshes <folder>), and takes any pictures and seeds you choose (--probes).

Verdict
    matches          at least the share of meshes the fingerprint states within its shape, box (and color) limits,
                     and the typical (median) shape distance within its limit
    DOES NOT MATCH   anything else
    signed           every answer carries a valid receipt from EmpirioLabs that names this model, and the mesh checked
                     is the file that answer lists
Exit status: 0 matches and signed, 1 anything else.
"""
import argparse
import array
import base64
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import check_image
import check_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINTS = os.path.join(HERE, "fingerprints")
KIND = "mesh"
ENDPOINT = "/v1/3d/generations"
ORIGIN = "%s://%s" % urllib.parse.urlsplit(check_receipt.API)[:2]
GRID = 16          # cells along each axis of the shape grid
COLOR_GRID = 8     # cells along each axis of the color grid (each holds 2 x 2 x 2 shape cells)
MAX_SPLIT = 32     # a triangle is cut into at most MAX_SPLIT ** 2 pieces, each placed in the cell of its center
JOB_WAIT_SECONDS = 4200
POLL_SECONDS = 5
RUNNING = ("queued", "processing", "running", "pending", "in_progress")
MADE_WITH = ("the pipeline published with the weights, on cuda", "the pipeline published with the weights, on cpu")
GLB_MAGIC = b"glTF"
CHUNK_JSON, CHUNK_BIN = 0x4E4F534A, 0x004E4942
COMPONENTS = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NORMALIZED = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
UNREAD_EXTENSIONS = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")


# --------------------------------------------------------------------------------------------------------- GLB
class NotAMesh(ValueError):
    """The bytes are not a GLB this module reads, or it holds no surface."""


def parse_glb(data):
    """(the glTF JSON, the binary chunk) of a GLB file."""
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise NotAMesh("not a binary glTF (GLB) file")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise NotAMesh("a glTF version this check does not read (%d)" % version)
    end, position, gltf, binary = min(length, len(data)), 12, None, b""
    while position + 8 <= end:
        size, kind = struct.unpack_from("<II", data, position)
        body = data[position + 8:position + 8 + size]
        if kind == CHUNK_JSON and gltf is None:
            try:
                gltf = json.loads(body.decode("utf-8").rstrip(" \t\r\n\x00"))
            except ValueError:
                raise NotAMesh("the GLB's JSON chunk is not JSON") from None
        elif kind == CHUNK_BIN and not binary:
            binary = body
        position += 8 + size + (-size % 4)
    if not isinstance(gltf, dict):
        raise NotAMesh("a GLB without its JSON chunk")
    unread = [name for name in gltf.get("extensionsRequired") or [] if name in UNREAD_EXTENSIONS]
    if unread:
        raise NotAMesh("a compressed mesh this check does not read (%s)" % ", ".join(unread))
    return gltf, binary


def _buffer(gltf, binary, index):
    buffer = (gltf.get("buffers") or [])[index]
    uri = buffer.get("uri")
    if uri is None:
        return binary
    if isinstance(uri, str) and uri.startswith("data:") and ";base64," in uri:
        return base64.b64decode(uri.split(",", 1)[1])
    raise NotAMesh("the mesh's data is in another file, which this check does not read")


def _view(gltf, binary, index):
    view = gltf["bufferViews"][index]
    data = _buffer(gltf, binary, view.get("buffer", 0))
    start = view.get("byteOffset", 0)
    if start + view["byteLength"] > len(data):
        raise NotAMesh("a buffer view runs past its buffer")
    return data[start:start + view["byteLength"]], view.get("byteStride")


def _elements(raw, offset, stride, count, code, width):
    """`count` elements of `width` numbers of component `code`, `stride` bytes apart from `offset`."""
    letter, size = COMPONENTS[code]
    element = size * width
    stride = stride or element
    if count and offset + stride * (count - 1) + element > len(raw):
        raise NotAMesh("an accessor runs past its buffer view")
    if stride == element:
        values = array.array(letter)
        values.frombytes(bytes(raw[offset:offset + element * count]))
        if sys.byteorder != "little":
            values.byteswap()
        return [tuple(values[i:i + width]) for i in range(0, len(values), width)]
    unpack = struct.Struct("<" + letter * width).unpack_from
    return [unpack(raw, offset + i * stride) for i in range(count)]


def accessor(gltf, binary, index):
    """An accessor's values: a list of tuples (of floats where the accessor is normalized)."""
    item = gltf["accessors"][index]
    code, width, count = item["componentType"], WIDTHS.get(item["type"]), item["count"]
    if code not in COMPONENTS or width is None:
        raise NotAMesh("an accessor of a type this check does not read")
    if "bufferView" in item:
        raw, stride = _view(gltf, binary, item["bufferView"])
        values = _elements(raw, item.get("byteOffset", 0), stride, count, code, width)
    else:
        values = [(0,) * width] * count
    sparse = item.get("sparse")
    if sparse:
        indices = sparse["indices"]
        raw, _ = _view(gltf, binary, indices["bufferView"])
        where = _elements(raw, indices.get("byteOffset", 0), None, sparse["count"], indices["componentType"], 1)
        raw, _ = _view(gltf, binary, sparse["values"]["bufferView"])
        new = _elements(raw, sparse["values"].get("byteOffset", 0), None, sparse["count"], code, width)
        values = list(values)
        for (i,), value in zip(where, new):
            if i >= count:
                raise NotAMesh("a sparse accessor names an element it does not have")
            values[i] = value
    if item.get("normalized") and code in NORMALIZED:
        scale = NORMALIZED[code]
        values = [tuple(max(v / scale, -1.0) for v in value) for value in values]
    return values


def _identity():
    return [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]


def _product(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def node_matrix(node):
    """A node's local transform: its column-major `matrix`, or translation x rotation (a quaternion) x scale."""
    if "matrix" in node:
        m = [float(v) for v in node["matrix"]]
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    tx, ty, tz = node.get("translation", (0, 0, 0))
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    sx, sy, sz = node.get("scale", (1, 1, 1))
    rotation = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    out = [[rotation[r][c] * (sx, sy, sz)[c] for c in range(3)] + [(tx, ty, tz)[r]] for r in range(3)]
    return out + [[0.0, 0.0, 0.0, 1.0]]


def mesh_instances(gltf):
    """(mesh index, world transform) of every mesh the scene shows; every mesh once when the file names no scene."""
    nodes = gltf.get("nodes") or []
    scenes = gltf.get("scenes") or []
    if not scenes:
        return [(index, _identity()) for index in range(len(gltf.get("meshes") or []))]
    found, stack = [], [(n, _identity(), 0) for n in scenes[gltf.get("scene", 0)].get("nodes") or []]
    while stack:
        index, parent, depth = stack.pop()
        if depth > 64:
            raise NotAMesh("the scene's nodes nest too deeply")
        node = nodes[index]
        world = _product(parent, node_matrix(node))
        if "mesh" in node:
            found.append((node["mesh"], world))
        stack.extend((child, world, depth + 1) for child in node.get("children") or [])
    return found


def _texture(gltf, binary, material):
    """(width, height, rows, factor, texcoord set) of a material's PNG base color texture, or None."""
    pbr = (material or {}).get("pbrMetallicRoughness") or {}
    factor = [float(v) for v in pbr.get("baseColorFactor", (1, 1, 1, 1))][:3]
    reference = pbr.get("baseColorTexture")
    if not reference:
        return None
    source = gltf["textures"][reference["index"]].get("source")
    image = (gltf.get("images") or [])[source] if source is not None else None
    if not image or "bufferView" not in image:
        return None
    raw, _ = _view(gltf, binary, image["bufferView"])
    try:
        width, height, rows = check_image.decode_png(bytes(raw))
    except check_image.NotAnImage:
        return None  # a JPEG or WebP texture: the colors are not read
    return width, height, rows, factor, reference.get("texCoord", 0)


def primitives(data, colors=True):
    """Every triangle primitive of a GLB, as (world positions, triangles as index triples, texture coordinates or
    None, its texture as _texture gives it or None)."""
    gltf, binary = parse_glb(data)
    out, decoded = [], {}
    for mesh_index, world in mesh_instances(gltf):
        for primitive in gltf["meshes"][mesh_index].get("primitives") or []:
            mode = primitive.get("mode", 4)
            attributes = primitive.get("attributes") or {}
            if mode not in (4, 5, 6) or "POSITION" not in attributes:
                continue
            local = accessor(gltf, binary, attributes["POSITION"])
            positions = [(world[0][0] * x + world[0][1] * y + world[0][2] * z + world[0][3],
                          world[1][0] * x + world[1][1] * y + world[1][2] * z + world[1][3],
                          world[2][0] * x + world[2][1] * y + world[2][2] * z + world[2][3]) for x, y, z in local]
            order = ([i for (i,) in accessor(gltf, binary, primitive["indices"])] if "indices" in primitive
                     else list(range(len(positions))))
            if mode == 4:
                triangles = [tuple(order[i:i + 3]) for i in range(0, len(order) - 2, 3)]
            elif mode == 5:
                triangles = [(order[i], order[i + 1], order[i + 2]) if i % 2 == 0 else
                             (order[i + 1], order[i], order[i + 2]) for i in range(len(order) - 2)]
            else:
                triangles = [(order[0], order[i], order[i + 1]) for i in range(1, len(order) - 1)]
            if any(max(t) >= len(positions) for t in triangles):
                raise NotAMesh("a triangle names a vertex the mesh does not have")
            texture, uvs = None, None
            material = primitive.get("material")
            if colors and material is not None:
                if material not in decoded:
                    decoded[material] = _texture(gltf, binary, gltf["materials"][material])
                texture = decoded[material]
                name = "TEXCOORD_%d" % texture[4] if texture else None
                if name in attributes:
                    uvs = accessor(gltf, binary, attributes[name])
                else:
                    texture = None
            out.append((positions, triangles, uvs, texture))
    if not out:
        raise NotAMesh("the GLB holds no triangles")
    return out


# ---------------------------------------------------------------------------------------------------- signature
def _texel_index(coordinate, size):
    """The texel a texture coordinate falls in: inside 0 to 1 (both ends included) the texture itself, outside it
    the texture repeated, as glTF's default sampler repeats it."""
    if 0.0 <= coordinate <= 1.0:
        return min(size - 1, int(coordinate * size))
    return int(math.floor(coordinate * size)) % size


def _texel(texture, u, v):
    width, height, rows, factor, _ = texture
    x = _texel_index(u, width)
    y = _texel_index(v, height)
    row = rows[y]
    return tuple(row[3 * x + c] * factor[c] for c in range(3))


def signature(data, colors=True):
    """{"box", "shape", "colors", "faces"} of a GLB: the box's proportions, the share of the surface in each of the
    GRID ** 3 cells (one byte a cell, 255 the fullest, as the square root of its share of the fullest cell), and the
    average base color of each COLOR_GRID ** 3 cell (three bytes a cell, 0 where it has no surface), or None without a
    texture this check reads."""
    parts = primitives(data, colors)
    low = [min(p[axis] for positions, *_ in parts for p in positions) for axis in range(3)]
    high = [max(p[axis] for positions, *_ in parts for p in positions) for axis in range(3)]
    sides = [high[axis] - low[axis] for axis in range(3)]
    longest = max(sides)
    if not longest > 0:
        raise NotAMesh("the mesh has no extent")
    center = [(low[axis] + high[axis]) / 2 for axis in range(3)]
    area = [0.0] * GRID ** 3
    paint = [[0.0, 0.0, 0.0, 0.0] for _ in range(COLOR_GRID ** 3)]
    any_paint, faces = False, 0
    step = longest / (2 * GRID)  # pieces at most half a cell across

    def cell(point, cells):
        index = 0
        for axis in range(3):
            g = int(((point[axis] - center[axis]) / longest + 0.5) * cells)
            index = index * cells + min(cells - 1, max(0, g))
        return index

    for positions, triangles, uvs, texture in parts:
        for a, b, c in triangles:
            pa, pb, pc = positions[a], positions[b], positions[c]
            ab = [pb[i] - pa[i] for i in range(3)]
            ac = [pc[i] - pa[i] for i in range(3)]
            cross = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
            twice = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
            if not twice > 0:
                continue
            faces += 1
            edge = max(math.dist(pa, pb), math.dist(pb, pc), math.dist(pa, pc))
            k = min(MAX_SPLIT, max(1, int(math.ceil(edge / step))))
            piece = twice / 2 / (k * k)
            ua = uvs[a] if uvs else None
            for i in range(k):
                for j in range(k - i):
                    centers = [((i + 1 / 3) / k, (j + 1 / 3) / k)]
                    if i + j < k - 1:
                        centers.append(((i + 2 / 3) / k, (j + 2 / 3) / k))
                    for s, t in centers:
                        point = [pa[n] + s * ab[n] + t * ac[n] for n in range(3)]
                        area[cell(point, GRID)] += piece
                        if ua is not None:
                            ub, uc = uvs[b], uvs[c]
                            color = _texel(texture, ua[0] + s * (ub[0] - ua[0]) + t * (uc[0] - ua[0]),
                                           ua[1] + s * (ub[1] - ua[1]) + t * (uc[1] - ua[1]))
                            box = paint[cell(point, COLOR_GRID)]
                            box[0] += piece * color[0]
                            box[1] += piece * color[1]
                            box[2] += piece * color[2]
                            box[3] += piece
                            any_paint = True
    if not faces:
        raise NotAMesh("the mesh's triangles have no area")
    fullest = max(area)
    shape = bytes(int(round(255 * math.sqrt(value / fullest))) for value in area)
    shown = None
    if any_paint:
        shown = bytes(min(255, int(round(channel / box[3]))) if box[3] > 0 else 0
                      for box in paint for channel in box[:3])
    return {"box": [round(side / longest, 4) for side in sides], "shape": shape, "colors": shown, "faces": faces}


def encode(values):
    return base64.b64encode(bytes(values)).decode("ascii")


def decode(text):
    return list(base64.b64decode(text))


def _colored_cells(shape):
    """The COLOR_GRID ** 3 cells that hold any surface, from the shape grid."""
    ratio = GRID // COLOR_GRID
    out = set()
    for index, value in enumerate(shape):
        if value:
            x, rest = divmod(index, GRID * GRID)
            y, z = divmod(rest, GRID)
            out.add(((x // ratio) * COLOR_GRID + y // ratio) * COLOR_GRID + z // ratio)
    return out


def distances(reference, live):
    """{"shape", "box", "color"} of a live signature against a fingerprint probe (color None when either has none)."""
    shape_a, shape_b = decode(reference["shape"]), list(live["shape"])
    gaps = {"shape": check_image.distance(shape_a, shape_b),
            "box": max(abs(a - b) for a, b in zip(reference["box"], live["box"])), "color": None}
    if reference.get("colors") and live.get("colors"):
        colors_a, colors_b = decode(reference["colors"]), list(live["colors"])
        common = sorted(_colored_cells(shape_a) & _colored_cells(shape_b))
        if len(common) >= 8:
            gaps["color"] = check_image.distance([colors_a[3 * c + n] for c in common for n in range(3)],
                                                 [colors_b[3 * c + n] for c in common for n in range(3)])
    return gaps


def verdict(fingerprint, lives):
    limits = fingerprint["limits"]
    color_limit = limits.get("color_distance")
    rows, close = [], 0
    for probe, live in zip(fingerprint["probes"], lives):
        gaps = distances(probe, live)
        ok = gaps["shape"] <= limits["shape_distance"] and gaps["box"] <= limits["box_gap"]
        if color_limit is not None and probe.get("colors"):
            ok = ok and gaps["color"] is not None and gaps["color"] <= color_limit
        close += ok
        rows.append(gaps)
    typical = statistics.median(row["shape"] for row in rows) if rows else float("inf")
    passed = bool(rows) and close / len(rows) >= limits["close"] and typical <= limits["typical_distance"]
    return {"result": "matches" if passed else "DOES NOT MATCH", "meshes": len(rows), "close": close,
            "typical_distance": typical, "distances": rows}


# ------------------------------------------------------------------------------------------------------ asking
class Unanswered(Exception):
    """The API gave no mesh for a probe: the check cannot be done, which says nothing about the model."""


def _send(method, path, key, body=None):
    sent = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(ORIGIN + path, data=sent, method=method,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "User-Agent": check_receipt.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600 if method == "POST" else 120) as response:
            received = response.read()
            return response.status, json.loads(received), sent, received
    except urllib.error.HTTPError as error:
        raise Unanswered("HTTP %d: %s" % (error.code, error.read()[:300].decode("utf-8", "replace"))) from None
    except (OSError, ValueError) as error:
        raise Unanswered("%s: %s" % (type(error).__name__, error)) from None


def probe_image(folder, image):
    """The bytes of a probe's picture beside its fingerprint, refused unless they are the file it records."""
    with open(os.path.join(folder, *str(image["file"]).split("/")), "rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise ValueError("the picture %s is not the file this fingerprint records" % image["file"])
    return data


def ask(model, probe, request, key, folder, exchanges=None, call=None):
    """One probe's GLB. A mesh still being made answers with a job, which is polled on the API's own origin to its end.
    A list `exchanges` gets (the request bytes, the bytes of the answer that carries the mesh, the mesh) for the
    receipt check. `call(method, path, key, body=None)` sends one request and returns (the HTTP status, the JSON
    answer, the request bytes, the answer bytes), as _send does, the default."""
    call = call or _send
    picture = probe_image(folder, probe["image"])
    body = {"model": model, **request, "seed": probe["seed"],
            "image": "data:image/png;base64," + base64.b64encode(picture).decode("ascii")}
    status, data, sent, received = call("POST", ENDPOINT, key, body)
    started = time.time()
    while status == 202 or (data.get("status") in RUNNING and not data.get("data")):
        if time.time() - started > JOB_WAIT_SECONDS:
            raise Unanswered("the 3D job did not finish in %d minutes" % (JOB_WAIT_SECONDS // 60))
        time.sleep(POLL_SECONDS)
        poll = data.get("poll_url")
        if not (isinstance(poll, str) and poll.startswith("/v1/jobs/")):
            poll = "/v1/jobs/%s" % urllib.parse.quote(str(data.get("job_id") or data.get("id") or ""), safe="")
        status, job, _sent, received = call("GET", poll, key)
        if job.get("status") in ("completed", "succeeded"):
            data = job.get("result") or {}
            break
        if job.get("status") not in RUNNING and job.get("status") is not None:
            raise Unanswered("the 3D job ended %s: %s" % (job.get("status"), json.dumps(job.get("error"))[:300]))
        data, status = job, 202
    items = [item for item in data.get("data") or [] if isinstance(item, dict)]
    if not items:
        raise Unanswered("the answer holds no mesh")
    if items[0].get("b64_json"):
        mesh = base64.b64decode(items[0]["b64_json"])
    elif items[0].get("url"):
        mesh = check_receipt.download(items[0]["url"])
    else:
        raise Unanswered("the answer's mesh has neither data nor a link")
    if exchanges is not None:
        exchanges.append((sent, received, mesh))
    return mesh


def load(path):
    with open(path, encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    if fingerprint.get("kind") != KIND:
        raise ValueError("%s: not a %s fingerprint" % (os.path.basename(path), KIND))
    return fingerprint


def run_check(model, fingerprint, key, keys=None, folder=FINGERPRINTS, ask_one=None, call=None, exchanges=None):
    """Every probe sent once; (the verdict, each live signature). With `keys` every answer's receipt is checked, and
    that the mesh is the file the answer lists (the verdict's `receipts`). `call` sends the requests through the
    caller's own transport, a list `exchanges` gets each (request bytes, answer bytes, mesh) for a caller that checks
    the receipts itself, and `ask_one(probe)` replaces the request to the API (tests)."""
    own = [] if (keys is not None or exchanges is not None) and ask_one is None else None
    colors = fingerprint["limits"].get("color_distance") is not None
    lives = []
    for number, probe in enumerate(fingerprint["probes"], 1):
        try:
            mesh = ask_one(probe) if ask_one else ask(model, probe, fingerprint["request"], key, folder, own, call)
            lives.append(signature(mesh, colors=colors))
        except (Unanswered, NotAMesh, check_image.NotAnImage) as error:
            raise Unanswered("the API did not answer mesh %d: %s" % (number, error)) from None
    v = verdict(fingerprint, lives)
    if exchanges is not None and own is not None:
        exchanges.extend(own)
    if keys is not None and own is not None:
        receipts = []
        for sent, received, mesh in own:
            ok, result = check_receipt.check(sent, received, keys)
            if ok:
                good, why = check_receipt.check_outputs(received, [mesh])
                if not good:
                    ok, result = False, why
            receipts.append((ok, result))
        v["receipts"] = receipts
    return v, lives


def report(model, fingerprint, v, seconds):
    limits = fingerprint["limits"]
    print("%s against the fingerprint of %s: %s" % (model, fingerprint["weights"], v["result"]))
    within = "shape %.3f and box %.3f" % (limits["shape_distance"], limits["box_gap"])
    if limits.get("color_distance") is not None:
        within += " and color %.3f" % limits["color_distance"]
    print("  meshes within %s of their signature: %d of %d (needs %d)" % (
        within, v["close"], v["meshes"], int(limits["close"] * v["meshes"] + 0.999)))
    print("  typical shape distance: %.4f (limit %.3f)" % (v["typical_distance"], limits["typical_distance"]))
    signed = check_receipt.report_receipts(model, v.get("receipts"))
    print("  %.0f s, no model was run" % seconds)
    return signed


def check_one(model, path, key, fingerprint=None, out="", keys=None):
    started = time.time()
    try:
        fingerprint = fingerprint or load(path)
        v, lives = run_check(model, fingerprint, key, keys=keys, folder=os.path.dirname(os.path.abspath(path)))
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print("%s could not be checked: %s" % (model, error))
        return 1
    signed = report(model, fingerprint, v, time.time() - started)
    if out:
        record = {"model": model, "fingerprint_of": fingerprint["weights"],
                  **{name: v[name] for name in v if name != "receipts"},
                  "signatures": [_shown(live) for live in lives]}
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=1, default=str)
    return 0 if v["result"] == "matches" and signed is not False else 1


def _shown(made):
    return {"box": made["box"], "faces": made["faces"], "shape": encode(made["shape"]),
            **({"colors": encode(made["colors"])} if made["colors"] else {})}


# -------------------------------------------------------------------------------------------------------- make
# The model's published parameter defaults (GET /v1/models lists them): what a request that leaves a setting out gets.
TRELLIS_DEFAULTS = {
    "pipeline_type": "1024_cascade", "preprocess_image": True, "max_num_tokens": 49152,
    "ss_sampling_steps": 12, "ss_guidance_strength": 7.5, "ss_guidance_rescale": 0.7,
    "ss_guidance_interval_start": 0.6, "ss_guidance_interval_end": 1.0, "ss_rescale_t": 5.0,
    "shape_slat_sampling_steps": 12, "shape_slat_guidance_strength": 7.5, "shape_slat_guidance_rescale": 0.5,
    "shape_slat_guidance_interval_start": 0.6, "shape_slat_guidance_interval_end": 1.0, "shape_slat_rescale_t": 3.0,
    "tex_slat_sampling_steps": 12, "tex_slat_guidance_strength": 1.0, "tex_slat_guidance_rescale": 0.0,
    "tex_slat_guidance_interval_start": 0.6, "tex_slat_guidance_interval_end": 0.9, "tex_slat_rescale_t": 3.0,
    "texture_size": 2048, "decimation_target": 500000, "remesh": True, "remesh_band": 1.0, "remesh_project": 0.9,
    "uv_unwrap_angle_threshold_deg": 90.0, "uv_unwrap_refine_iterations": 0, "uv_unwrap_global_iterations": 1,
    "uv_unwrap_smooth_strength": 1.0,
}
# Where the TRELLIS.2 pipeline reads each component repository from (its pipeline.json), by --component role.
TRELLIS_COMPONENTS = {"sparse structure decoder": ("models", "sparse_structure_decoder"),
                      "image encoder": ("image_cond_model", "model_name"),
                      "background remover": ("rembg_model", "model_name")}


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _sampler(settings, prefix):
    return {"steps": int(settings[prefix + "_sampling_steps"]),
            "guidance_strength": float(settings[prefix + "_guidance_strength"]),
            "guidance_rescale": float(settings[prefix + "_guidance_rescale"]),
            "guidance_interval": [float(settings[prefix + "_guidance_interval_start"]),
                                  float(settings[prefix + "_guidance_interval_end"])],
            "rescale_t": float(settings[prefix + "_rescale_t"])}


def _trellis2(weights, components, device):
    """TRELLIS.2 (the code published with microsoft/TRELLIS.2-4B), each component read from the local copy of the
    repository its --component names, as the pipeline's own pipeline.json names them: the function that makes a GLB."""
    import o_voxel
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    with open(os.path.join(weights, "pipeline.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    for item in components:
        place = TRELLIS_COMPONENTS.get(item["role"])
        if place and item.get("path"):
            section, name = place
            if section == "models":
                config["args"]["models"][name] = os.path.join(item["path"], "ckpts", os.path.basename(
                    config["args"]["models"][name]))
            else:
                config["args"].setdefault(section, {}).setdefault("args", {})[name] = item["path"]
    with open(os.path.join(weights, "pipeline.verifiable.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(weights, config_file="pipeline.verifiable.json")
    pipe.cuda() if device.startswith("cuda") else pipe.to(device)

    def make_glb(probe, request, image_path):
        settings = {**TRELLIS_DEFAULTS, **request}
        mesh = pipe.run(Image.open(image_path).convert("RGBA"), num_samples=1, seed=int(probe["seed"]),
                        sparse_structure_sampler_params=_sampler(settings, "ss"),
                        shape_slat_sampler_params=_sampler(settings, "shape_slat"),
                        tex_slat_sampler_params=_sampler(settings, "tex_slat"),
                        preprocess_image=bool(settings["preprocess_image"]),
                        pipeline_type=str(settings["pipeline_type"]),
                        max_num_tokens=int(settings["max_num_tokens"]))[0]
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(settings["decimation_target"]), texture_size=int(settings["texture_size"]),
            remesh=bool(settings["remesh"]), remesh_band=float(settings["remesh_band"]),
            remesh_project=float(settings["remesh_project"]),
            mesh_cluster_threshold_cone_half_angle_rad=_float32(math.radians(
                float(settings["uv_unwrap_angle_threshold_deg"]))),
            mesh_cluster_refine_iterations=int(settings["uv_unwrap_refine_iterations"]),
            mesh_cluster_global_iterations=int(settings["uv_unwrap_global_iterations"]),
            mesh_cluster_smooth_strength=float(settings["uv_unwrap_smooth_strength"]))
        folder = tempfile.mkdtemp(prefix="verifiable-inference-")
        try:
            path = os.path.join(folder, "mesh.glb")
            glb.export(path, extension_webp=False)
            with open(path, "rb") as handle:
                return handle.read()
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return make_glb


def _components(specs):
    """Each --component role=...,weights=<repo>,revision=<commit>[,path=<local copy>]: a public repository the mesh was
    made from besides the weights."""
    out = []
    for spec in specs or []:
        fields = dict(part.split("=", 1) for part in spec.split(","))
        out.append({"role": fields["role"], "weights": fields["weights"], "revision": fields["revision"],
                    **({"path": fields["path"]} if fields.get("path") else {})})
    return out


def _recorded_image(image, model, out_folder):
    name = os.path.basename(image["path"])
    target = os.path.join(out_folder, model)
    os.makedirs(target, exist_ok=True)
    shutil.copyfile(image["path"], os.path.join(target, name))
    recorded = {"file": "%s/%s" % (model, name), "sha256": check_image.sha256_of(image["path"]),
                "source": image["source"], "license": image["license"]}
    if image.get("processing"):
        recorded["processing"] = image["processing"]
    return recorded


def make(args):
    with open(args.probes, encoding="utf-8") as handle:
        probes = json.load(handle)
    request = json.loads(args.request)
    components = _components(args.component)
    out_folder = os.path.dirname(os.path.abspath(args.out))
    make_glb = None if args.meshes else _trellis2(args.weights, components, args.device)
    colors = args.color_distance is not None
    rows, started = [], time.time()
    for probe in probes:
        t = time.time()
        if make_glb is None:
            with open(os.path.join(args.meshes, "%d.glb" % probe["seed"]), "rb") as handle:
                mesh = handle.read()
        else:
            mesh = make_glb(probe, request, probe["image"]["path"])
        made = signature(mesh, colors=colors)
        rows.append({"image": _recorded_image(probe["image"], args.model, out_folder), "seed": probe["seed"],
                     **_shown(made)})
        print("%-40s seed %-10d %d faces %.0f s" % (os.path.basename(probe["image"]["path"]), probe["seed"],
                                                   made["faces"], time.time() - t), flush=True)
    limits = {"shape_distance": args.shape_distance, "typical_distance": args.typical_distance,
              "box_gap": args.box_gap, "close": args.close}
    if colors:
        limits["color_distance"] = args.color_distance
    fingerprint = {"kind": KIND, "model": args.model, "weights": args.name, "revision": args.revision,
                   "weight_files_sha256": check_image._weight_files(args.weights),
                   "made_with": "the pipeline published with the weights, on %s" % args.device.split(":")[0],
                   "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": request,
                   "limits": limits, "probes": rows}
    if components:
        fingerprint["components"] = [{key: item[key] for key in ("role", "weights", "revision")} for item in components]
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(fingerprint, handle, ensure_ascii=False, indent=1)
    print("fingerprint of %d meshes written in %.0f s" % (len(rows), time.time() - started))


def _signature_command(args):
    with open(args.mesh, "rb") as handle:
        print(json.dumps(_shown(signature(handle.read())), indent=1))
    return 0


def _compare_command(args):
    with open(args.mesh, "rb") as handle:
        first = _shown(signature(handle.read()))
    with open(args.other, "rb") as handle:
        second = signature(handle.read())
    gaps = distances(first, second)
    print("shape %.4f, box %.4f, color %s" % (gaps["shape"], gaps["box"],
                                              "not read" if gaps["color"] is None else "%.4f" % gaps["color"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a live model against its fingerprint, as check_model.py does")
    c.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI, for example trellis-2-4b")
    c.add_argument("--fingerprint", default="", help="a fingerprint file (default: fingerprints/<model>.json)")
    c.add_argument("--out", default="", help="also write the numbers read from each answer to this JSON file")
    c.add_argument("--keys", default="pinned", help="receipt keys: pinned (default), live, a keys JSON file, or none")
    s = sub.add_parser("signature", help="print the signature of a GLB you have")
    s.add_argument("mesh")
    p = sub.add_parser("compare", help="how far two GLBs are")
    p.add_argument("mesh")
    p.add_argument("other")
    m = sub.add_parser("make", help="make a fingerprint from public weights")
    m.add_argument("--weights", required=True, help="a local folder holding the public checkpoint (its files are hashed)")
    m.add_argument("--name", required=True, help="the public repository these weights are")
    m.add_argument("--revision", required=True, help="the repository revision")
    m.add_argument("--model", required=True, help="the model ID on EmpirioLabs AI")
    m.add_argument("--probes", required=True,
                   help="a JSON list of {image: {path, source, license, processing}, seed}: any pictures and seeds")
    m.add_argument("--request", required=True, help="the API settings every mesh is asked with (JSON)")
    m.add_argument("--component", action="append",
                   help="role=<role>,weights=<repo>,revision=<commit>[,path=<local copy>] for each other repository")
    source = m.add_mutually_exclusive_group(required=True)
    source.add_argument("--pipeline", choices=("trellis2",), help="make each mesh in this process")
    source.add_argument("--meshes", help="a folder of <seed>.glb meshes another public program made")
    m.add_argument("--device", default="cuda")
    m.add_argument("--out", required=True)
    m.add_argument("--shape-distance", type=float, required=True)
    m.add_argument("--typical-distance", type=float, required=True)
    m.add_argument("--box-gap", type=float, required=True)
    m.add_argument("--color-distance", type=float, default=None)
    m.add_argument("--close", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make":
            return make(args)
        if args.command == "signature":
            return _signature_command(args)
        if args.command == "compare":
            return _compare_command(args)
    except NotAMesh as error:
        print(error)
        return 1
    return check_one(args.model, args.fingerprint or os.path.join(FINGERPRINTS, args.model + ".json"),
                     check_receipt.api_key(), out=args.out,
                     keys=None if args.keys == "none" else check_receipt.load_keys(args.keys))


if __name__ == "__main__":
    sys.exit(main() or 0)
