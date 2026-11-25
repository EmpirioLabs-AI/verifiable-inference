"""Offline tests of check_mesh.py: the standard-library GLB reader reads interleaved, normalized and sparse accessors,
triangle strips and fans, node transforms and a PNG base color texture, and refuses what it cannot read; the signature
survives remeshing, decimation, a move or a scale of the whole mesh and another texture layout, and tells another shape
or other colors apart; a model's own meshes match, another model's do not, a picture that is not the recorded file is
never sent, every answer's receipt is checked, check_model.py runs this check for a mesh fingerprint, and the check
needs only the standard library. Run with: python -m unittest discover -s tests"""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import check_mesh as cm  # noqa: E402
import check_model  # noqa: E402
from test_check_image import png  # noqa: E402  (the PNG writer)
from test_check_receipt import KEYS, receipt_for  # noqa: E402  (the test signer, never a key of the API)

FINGERPRINTS = os.path.join(ROOT, "fingerprints")


# --------------------------------------------------------------------------------------------------- a GLB writer
def glb(positions, triangles, uvs=None, texture=None, node=None, quantize=False, interleave=False, sparse=None,
        mode=4, required=None, indices16=False):
    """A GLB of one mesh: positions (x, y, z), triangles as index triples (or one index list for a strip or fan),
    optional texture coordinates and a PNG base color texture, an optional node transform."""
    blob = bytearray()
    views, accessors = [], []

    def view(data, stride=None):
        blob.extend(b"\x00" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data),
                      **({"byteStride": stride} if stride else {})})
        blob.extend(data)
        return len(views) - 1

    if quantize:
        scale = max(abs(v) for p in positions for v in p)
        data = b"".join(struct.pack("<hhh", *(int(round(v / scale * 32767)) for v in p)) + b"\x00\x00"
                        for p in positions)
        accessors.append({"bufferView": view(data, 8), "componentType": 5122, "normalized": True,
                          "count": len(positions), "type": "VEC3"})
        node = node or {"scale": [scale, scale, scale]}
    elif interleave:
        data = b"".join(struct.pack("<3f", *p) + struct.pack("<f", 9.0) for p in positions)
        accessors.append({"bufferView": view(data, 16), "componentType": 5126, "count": len(positions), "type": "VEC3"})
    else:
        data = b"".join(struct.pack("<3f", *p) for p in positions)
        accessors.append({"bufferView": view(data), "componentType": 5126, "count": len(positions), "type": "VEC3"})
    if sparse:
        where, values = zip(*sparse)
        accessors[0]["sparse"] = {
            "count": len(where),
            "indices": {"bufferView": view(struct.pack("<%dI" % len(where), *where)), "componentType": 5125},
            "values": {"bufferView": view(b"".join(struct.pack("<3f", *v) for v in values))}}
    flat = list(triangles) if mode != 4 else [i for t in triangles for i in t]
    letter, code = ("H", 5123) if indices16 else ("I", 5125)
    accessors.append({"bufferView": view(struct.pack("<%d%s" % (len(flat), letter), *flat)), "componentType": code,
                      "count": len(flat), "type": "SCALAR"})
    primitive = {"attributes": {"POSITION": 0}, "indices": 1, "mode": mode}
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, **(node or {})}], "meshes": [{"primitives": [primitive]}], "accessors": accessors,
            "bufferViews": views}
    if uvs is not None:
        data = b"".join(struct.pack("<2f", *uv) for uv in uvs)
        accessors.append({"bufferView": view(data), "componentType": 5126, "count": len(uvs), "type": "VEC2"})
        primitive["attributes"]["TEXCOORD_0"] = len(accessors) - 1
    if texture is not None:
        gltf["images"] = [{"bufferView": view(texture), "mimeType": "image/png"}]
        gltf["textures"] = [{"source": 0}]
        gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        primitive["material"] = 0
    if required:
        gltf["extensionsRequired"] = required
    gltf["buffers"] = [{"byteLength": len(blob)}]
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob.extend(b"\x00" * (-len(blob) % 4))
    body = struct.pack("<II", len(text), cm.CHUNK_JSON) + text + struct.pack("<II", len(blob), cm.CHUNK_BIN) + bytes(blob)
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


def box(sx=1.0, sy=1.0, sz=1.0, n=1):
    """A box's surface, each face cut into n x n squares of two triangles: (positions, triangles, uvs with u the
    position along x and v along y)."""
    positions, triangles, uvs = [], [], []
    for axis in range(3):
        for side in (-0.5, 0.5):
            first, second = [a for a in range(3) if a != axis]
            base = len(positions)
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[axis], p[first], p[second] = side, i / n - 0.5, j / n - 0.5
                    positions.append((p[0] * sx, p[1] * sy, p[2] * sz))
                    uvs.append((p[0] + 0.5, p[1] + 0.5))
            for i in range(n):
                for j in range(n):
                    a, b = base + i * (n + 1) + j, base + (i + 1) * (n + 1) + j
                    triangles += [(a, b, a + 1), (b, b + 1, a + 1)]
    return positions, triangles, uvs


def sphere(rings=24, segments=48):
    positions, triangles = [], []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(segments):
            phi = 2 * math.pi * s / segments
            positions.append((0.5 * math.sin(theta) * math.cos(phi), 0.5 * math.sin(theta) * math.sin(phi),
                              0.5 * math.cos(theta)))
    for r in range(rings):
        for s in range(segments):
            a, b = r * segments + s, r * segments + (s + 1) % segments
            c, d = a + segments, b + segments
            triangles += [(a, c, b), (b, c, d)]
    return positions, triangles


def halves(left=(220, 30, 30), right=(30, 30, 220), flip=False):
    """A 64 x 64 PNG, one color on the left half and another on the right (swapped when flipped)."""
    if flip:
        left, right = right, left
    return png(64, 64, lambda x, y: left if x < 32 else right)


def fingerprint(meshes, colors=False):
    probes = []
    for i, data in enumerate(meshes):
        made = cm.signature(data, colors=colors)
        probes.append({"image": {"file": "test-mesh-model/p%d.png" % i, "sha256": "0" * 64,
                                 "source": "https://example.org/p", "license": "CC0"}, "seed": 100 + i,
                       **cm._shown(made)})
    limits = {"shape_distance": 0.1, "typical_distance": 0.05, "box_gap": 0.05, "close": 0.75}
    if colors:
        limits["color_distance"] = 0.1
    return {"kind": "mesh", "model": "test-mesh-model", "weights": "org/test-mesh-model", "revision": "a" * 40,
            "weight_files_sha256": {"ckpts/model.safetensors": "b" * 64},
            "made_with": "the pipeline published with the weights, on cuda", "made_at": "2026-11-25T00:00:00Z",
            "request": {"pipeline_type": "512", "texture_size": "1024", "decimation_target": 10000},
            "limits": limits, "probes": probes}


SHAPES = [box(1, 1, 1, 2), box(2, 1, 1, 2), box(1, 2, 0.5, 2), box(3, 1, 2, 2), sphere(12, 24), box(1, 0.3, 1, 2),
          box(2, 2, 1, 2), box(1, 3, 3, 2)]


def mesh_bytes(shape, **options):
    return glb(shape[0], shape[1], **options)


def run_with(fp, mesh_for, keys=None):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, fp["model"] + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fp, handle)

        def ask(model, probe, request, key, folder, exchanges=None, call=None):
            mesh = mesh_for(probe)
            if exchanges is not None:
                sent = json.dumps({"model": model, "seed": probe["seed"], **request}).encode()
                body = json.dumps({"data": [{"url": "https://media.example/m.glb",
                                             "sha256": hashlib.sha256(mesh).hexdigest()}]}).encode()
                receipt = json.dumps(receipt_for(sent, body, model=model), separators=(",", ":")).encode()
                exchanges.append((sent, body[:-1] + b',"receipt":' + receipt + b"}", mesh))
            return mesh
        out = io.StringIO()
        with mock.patch.object(cm, "ask", ask), contextlib.redirect_stdout(out):
            code = cm.check_one(fp["model"], path, "key", keys=keys)
        return code, out.getvalue()


class Reader(unittest.TestCase):
    def test_what_is_not_a_glb_or_is_compressed_is_refused(self):
        with self.assertRaises(cm.NotAMesh):
            cm.parse_glb(b"solid ascii stl file")
        with self.assertRaises(cm.NotAMesh):
            cm.parse_glb(b"glTF" + struct.pack("<II", 1, 20) + b"\x00" * 8)
        with self.assertRaises(cm.NotAMesh) as caught:
            cm.parse_glb(mesh_bytes(box(), required=["KHR_draco_mesh_compression"]))
        self.assertIn("compressed", str(caught.exception))

    def test_interleaved_normalized_and_sparse_positions_read_as_written(self):
        positions, triangles, _ = box(1, 2, 3)
        plain = cm.primitives(glb(positions, triangles))[0][0]
        self.assertEqual(cm.primitives(glb(positions, triangles, interleave=True))[0][0], plain)
        quantized = cm.primitives(glb(positions, triangles, quantize=True))[0][0]
        self.assertTrue(all(abs(a - b) < 1e-3 for p, q in zip(plain, quantized) for a, b in zip(p, q)))
        moved = cm.primitives(glb(positions, triangles, sparse=[(0, (9.0, 9.0, 9.0))]))[0][0]
        self.assertEqual(moved[0], (9.0, 9.0, 9.0))
        self.assertEqual(moved[1:], plain[1:])

    def test_strips_fans_and_short_indices_make_the_same_triangles(self):
        square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        strip = cm.primitives(glb(square, [0, 1, 2, 3], mode=5))[0][1]
        self.assertEqual(strip, [(0, 1, 2), (2, 1, 3)])
        fan = cm.primitives(glb(square, [0, 1, 3, 2], mode=6, indices16=True))[0][1]
        self.assertEqual(fan, [(0, 1, 3), (0, 3, 2)])

    def test_node_transforms_reach_the_positions(self):
        positions, triangles, _ = box()
        translated = cm.primitives(glb(positions, triangles, node={"translation": [10, 0, 0], "scale": [2, 2, 2]}))
        self.assertEqual(translated[0][0][0], (positions[0][0] * 2 + 10, positions[0][1] * 2, positions[0][2] * 2))
        matrix = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 0, 0, 1]  # column-major, the same transform
        self.assertEqual(cm.primitives(glb(positions, triangles, node={"matrix": matrix}))[0][0], translated[0][0])
        quarter = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]  # 90 degrees about z
        turned = cm.primitives(glb(positions, triangles, node={"rotation": quarter}))[0][0]
        x, y, z = positions[0]
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(turned[0], (-y, x, z))))

    def test_a_triangle_naming_a_missing_vertex_is_refused(self):
        with self.assertRaises(cm.NotAMesh):
            cm.primitives(glb([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 7)]))


class Signature(unittest.TestCase):
    def test_the_box_is_read_as_proportions_of_its_longest_side(self):
        made = cm.signature(mesh_bytes(box(2, 1, 0.5)))
        self.assertEqual(made["box"], [1.0, 0.5, 0.25])
        self.assertEqual(len(made["shape"]), cm.GRID ** 3)
        self.assertEqual(max(made["shape"]), 255)

    def test_the_shape_survives_remeshing_a_move_and_a_scale(self):
        coarse = cm.signature(mesh_bytes(box(2, 1, 1, 1)))
        fine = cm.signature(mesh_bytes(box(2, 1, 1, 9)))
        self.assertLess(cm.check_image.distance(list(coarse["shape"]), list(fine["shape"])), 0.01)
        positions, triangles, _ = box(2, 1, 1, 3)
        moved = cm.signature(glb(positions, triangles, node={"translation": [5, -3, 2], "scale": [4, 4, 4]}))
        self.assertLess(cm.check_image.distance(list(fine["shape"]), list(moved["shape"])), 0.01)
        self.assertEqual(moved["box"], fine["box"])

    def test_another_shape_is_far(self):
        cube = cm.signature(mesh_bytes(box(1, 1, 1, 4)))
        ball = cm.signature(mesh_bytes(sphere()))
        self.assertGreater(cm.check_image.distance(list(cube["shape"]), list(ball["shape"])), 0.3)

    def test_colors_survive_another_texture_layout_and_tell_other_colors_apart(self):
        positions, triangles, uvs = box(1, 1, 1, 3)
        made = cm.signature(glb(positions, triangles, uvs=uvs, texture=halves()))
        mirrored_uvs = [(1 - u, v) for u, v in uvs]
        relaid = cm.signature(glb(positions, triangles, uvs=mirrored_uvs, texture=halves(flip=True)))
        swapped = cm.signature(glb(positions, triangles, uvs=uvs, texture=halves(flip=True)))
        reference = cm._shown(made)
        self.assertIsNotNone(made["colors"])
        self.assertLess(cm.distances(reference, relaid)["color"], 0.01)
        self.assertGreater(cm.distances(reference, swapped)["color"], 0.5)
        self.assertIsNone(cm.signature(glb(positions, triangles, uvs=uvs, texture=halves()), colors=False)["colors"])


class Verdict(unittest.TestCase):
    def test_a_models_own_meshes_match_even_remeshed(self):
        fp = fingerprint([mesh_bytes(shape) for shape in SHAPES])
        finer = {100 + i: (box(*args, n=5) if i != 4 else sphere(16, 32))
                 for i, args in enumerate([(1, 1, 1), (2, 1, 1), (1, 2, 0.5), (3, 1, 2), None, (1, 0.3, 1),
                                           (2, 2, 1), (1, 3, 3)])}
        code, out = run_with(fp, lambda probe: mesh_bytes(finer[probe["seed"]]))
        self.assertEqual(code, 0, out)
        self.assertIn(": matches", out)

    def test_another_models_meshes_do_not_match(self):
        fp = fingerprint([mesh_bytes(shape) for shape in SHAPES])
        others = list(reversed(SHAPES))
        code, out = run_with(fp, lambda probe: mesh_bytes(others[probe["seed"] - 100]))
        self.assertEqual(code, 1)
        self.assertIn("DOES NOT MATCH", out)

    def test_a_mesh_of_other_proportions_is_not_close(self):
        fp = fingerprint([mesh_bytes(shape) for shape in SHAPES])
        stretched = {100 + i: shape for i, shape in enumerate(SHAPES)}
        stretched[100], stretched[101], stretched[102] = box(1, 1, 1.2, 2), box(2, 1, 1.25, 2), box(1, 2, 0.7, 2)
        code, out = run_with(fp, lambda probe: mesh_bytes(stretched[probe["seed"]]))
        self.assertEqual(code, 1, out)
        self.assertIn("5 of 8", out)

    def test_every_answer_signed_for_its_mesh_is_reported_signed(self):
        fp = fingerprint([mesh_bytes(shape) for shape in SHAPES])
        code, out = run_with(fp, lambda probe: mesh_bytes(SHAPES[probe["seed"] - 100]), keys=KEYS)
        self.assertEqual(code, 0, out)
        self.assertIn("every answer signed by EmpirioLabs for test-mesh-model", out)

    def test_with_a_color_limit_the_colors_decide_too(self):
        positions, triangles, uvs = box(1, 1, 1, 2)
        own = glb(positions, triangles, uvs=uvs, texture=halves())
        fp = fingerprint([own] * 4, colors=True)
        self.assertEqual(run_with(fp, lambda probe: own)[0], 0)
        repainted = glb(positions, triangles, uvs=uvs, texture=halves(flip=True))
        code, out = run_with(fp, lambda probe: repainted)
        self.assertEqual(code, 1, out)


class Asking(unittest.TestCase):
    def test_a_picture_that_is_not_the_recorded_file_is_never_sent(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "m"))
            with open(os.path.join(folder, "m", "p.png"), "wb") as handle:
                handle.write(b"other bytes")
            sent = []
            with self.assertRaises(ValueError):
                cm.ask("m", {"image": {"file": "m/p.png", "sha256": "0" * 64}, "seed": 1}, {}, "key", folder,
                       call=lambda *a: sent.append(a))
        self.assertEqual(sent, [])

    def test_a_job_is_polled_on_the_apis_own_origin_until_it_holds_the_mesh(self):
        mesh = mesh_bytes(box())
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "m"))
            picture = png(4, 4, lambda x, y: (1, 2, 3))
            with open(os.path.join(folder, "m", "p.png"), "wb") as handle:
                handle.write(picture)
            probe = {"image": {"file": "m/p.png", "sha256": hashlib.sha256(picture).hexdigest()}, "seed": 9}
            calls = []
            done = {"status": "completed", "result": {"data": [{"b64_json": __import__("base64").b64encode(mesh).decode()}]}}

            def call(method, path, key, body=None):
                calls.append((method, path))
                if method == "POST":
                    answer = {"status": "queued", "job_id": "j 1", "poll_url": "https://elsewhere.example/steal"}
                    return 202, answer, json.dumps(body).encode(), json.dumps(answer).encode()
                return 200, done, None, json.dumps(done).encode()
            exchanges = []
            with mock.patch.object(cm.time, "sleep", lambda s: None):
                got = cm.ask("m", probe, {"pipeline_type": "512"}, "key", folder, exchanges, call=call)
        self.assertEqual(got, mesh)
        self.assertEqual(calls, [("POST", cm.ENDPOINT), ("GET", "/v1/jobs/j%201")])
        sent = json.loads(exchanges[0][0])
        self.assertEqual((sent["seed"], sent["pipeline_type"]), (9, "512"))
        self.assertTrue(sent["image"].startswith("data:image/png;base64,"))


class Wiring(unittest.TestCase):
    def test_check_model_runs_this_check_for_a_mesh_fingerprint(self):
        self.assertIs(check_model.OTHER_KINDS["mesh"], cm)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test-mesh-model.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fingerprint([mesh_bytes(box())]), handle)
            with mock.patch.object(cm, "check_one", return_value=0) as one:
                self.assertEqual(check_model.check_one("test-mesh-model", path, "key", None), 0)
            one.assert_called_once()

    def test_the_check_needs_only_the_standard_library(self):
        with open(os.path.join(ROOT, "check_mesh.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                top.add((node.module or "").split(".")[0])
        allowed = set(getattr(sys, "stdlib_module_names", ())) | {"check_receipt", "check_image"}
        self.assertLessEqual(top, allowed, top - allowed)


if __name__ == "__main__":
    unittest.main()
