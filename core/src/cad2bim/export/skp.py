"""건물을 스케치업 파일(.skp)로 쓴다. 공식 C SDK의 DLL을 직접 호출한다.

요소마다 그룹 하나이고, 층은 태그 폴더, 종류는 태그다. SDK는 저장소에 넣지 않고 설정의 위치에서 읽는다.
공식 SDK가 없으면 건너뛴다. 이름은 층 코드와 요소 번호뿐이다.
"""
from __future__ import annotations

import atexit
import ctypes
import os
from ctypes import POINTER, byref, c_bool, c_char_p, c_double, c_int32, c_size_t, c_void_p
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aerial import choose_view
from .collada import _GROUPS, _VOLUME_TOLERANCE, MATERIALS, _collect
from .mesh import Mesh

_MM_PER_INCH = 25.4
_DLL = Path("binaries/sketchup/x64/SketchUpAPI.dll")
_UNIT_MILLIMETER = 2
_FORMAT_DECIMAL = 0
SU5_2005 = 2  # SUModelVersion_SU5. 헤더의 enum 순서다
_api: Any = None


class _Point(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class _Plane(ctypes.Structure):
    _fields_ = [("a", c_double), ("b", c_double), ("c", c_double), ("d", c_double)]


class _Color(ctypes.Structure):
    _fields_ = [("r", ctypes.c_ubyte), ("g", ctypes.c_ubyte), ("b", ctypes.c_ubyte), ("a", ctypes.c_ubyte)]


class _Point2(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class _MaterialInput(ctypes.Structure):
    _fields_ = [
        ("num_uv_coords", c_size_t), ("uv_coords", _Point2 * 4), ("vertex_indices", c_size_t * 4), ("material", c_void_p),
    ]


class SketchUpError(RuntimeError):
    pass


def find_sdk(cfg: dict[str, Any]) -> Path | None:
    for candidate in (os.environ.get("CAD2BIM_SKETCHUP_SDK"), cfg.get("sketchup_sdk_dir")):
        if candidate and (Path(candidate) / _DLL).is_file():
            return Path(candidate)
    return None


def _load(sdk: Path) -> Any:
    global _api
    if _api is None:
        directory = (sdk / _DLL).parent
        os.add_dll_directory(str(directory))
        _api = ctypes.CDLL(str(sdk / _DLL))
        _api.SUGroupToDrawingElement.restype = c_void_p
        _api.SUGroupToDrawingElement.argtypes = [c_void_p]
        _api.SUInitialize()
        atexit.register(_api.SUTerminate)
    return _api


def _ok(result: int, name: str) -> None:
    if result != 0:
        raise SketchUpError(f"{name}: {result}")


def _text(api: Any, getter: Any, *args: Any) -> str:
    handle = c_void_p()
    _ok(api.SUStringCreate(byref(handle)), "string")
    try:
        _ok(getter(*args, byref(handle)), "get string")
        length = c_size_t()
        _ok(api.SUStringGetUTF8Length(handle, byref(length)), "string length")
        buffer = ctypes.create_string_buffer(length.value + 1)
        written = c_size_t()
        _ok(api.SUStringGetUTF8(handle, c_size_t(length.value + 1), buffer, byref(written)), "string data")
        return buffer.value.decode("utf-8")
    finally:
        api.SUStringRelease(byref(handle))


def _set_units(api: Any, model: c_void_p) -> None:
    """가져온 사람이 mm로 보도록 단위를 맞춘다. 실패해도 형상은 그대로라 멈추지 않는다."""
    manager, provider, value = c_void_p(), c_void_p(), c_void_p()
    if api.SUModelGetOptionsManager(model, byref(manager)) or api.SUOptionsManagerGetOptionsProviderByName(
        manager, b"UnitsOptions", byref(provider)
    ):
        return
    if api.SUTypedValueCreate(byref(value)):
        return
    for name, number in ((b"LengthFormat", _FORMAT_DECIMAL), (b"LengthUnit", _UNIT_MILLIMETER)):
        if not api.SUTypedValueSetInt32(value, c_int32(number)):
            api.SUOptionsProviderSetValue(provider, c_char_p(name), value)
    api.SUTypedValueRelease(byref(value))


def _set_view(api: Any, model: c_void_p, view: Any) -> None:
    """파일을 열면 조감도 시점으로 보이게 하고, 같은 시점을 장면으로 남긴다. 실패해도 형상은 그대로다."""
    eye = _Point(*(view.eye / _MM_PER_INCH).tolist())
    target = _Point(*(view.target / _MM_PER_INCH).tolist())
    up = _Point(0.0, 0.0, 1.0)

    def aim(camera: c_void_p) -> int:
        return (
            api.SUCameraSetOrientation(camera, byref(eye), byref(target), byref(up))
            or api.SUCameraSetPerspective(camera, c_bool(True))
            or api.SUCameraSetPerspectiveFrustumFOV(camera, c_double(view.fov_degrees))
        )

    active, scene, camera = c_void_p(), c_void_p(), c_void_p()
    if not api.SUModelGetCamera(model, byref(active)):
        aim(active)
    if api.SUSceneCreate(byref(scene)) or api.SUSceneSetName(scene, "조감도".encode()) or api.SUCameraCreate(byref(camera)):
        return
    if aim(camera) or api.SUSceneSetCamera(scene, camera):
        return
    api.SUCameraRelease(byref(camera))
    index = c_int32()
    if not api.SUModelAddScene(model, c_int32(-1), scene, byref(index)):
        api.SUModelSetActiveScene(model, scene)
    options, value = c_void_p(), c_void_p()
    if not api.SUModelGetRenderingOptions(model, byref(options)) and not api.SUTypedValueCreate(byref(value)):
        if not api.SUTypedValueSetBool(value, c_bool(False)):  # 굵은 외곽선을 끈다
            api.SURenderingOptionsSetValue(options, c_char_p(b"DrawSilhouettes"), value)
        api.SUTypedValueRelease(byref(value))


def read_view(path: Path, sdk: Path) -> dict[str, Any]:
    """저장한 파일의 장면 수와 현재 카메라 위치(mm)를 읽는다."""
    api = _load(sdk)
    model, camera = c_void_p(), c_void_p()
    _ok(api.SUModelCreateFromFile(byref(model), str(path).encode("utf-8")), "open")
    try:
        count = c_size_t()
        _ok(api.SUModelGetNumScenes(model, byref(count)), "scenes")
        _ok(api.SUModelGetCamera(model, byref(camera)), "camera")
        eye, target, up = _Point(), _Point(), _Point()
        _ok(api.SUCameraGetOrientation(camera, byref(eye), byref(target), byref(up)), "orientation")
        return {"scenes": count.value, "eye_mm": (eye.x * _MM_PER_INCH, eye.y * _MM_PER_INCH, eye.z * _MM_PER_INCH)}
    finally:
        api.SUModelRelease(byref(model))


def _prepare_view(api: Any, model: c_void_p, collected: list[Any], cfg: dict[str, Any]) -> None:
    """같은 시점과 시각을 파일에 넣는다. 조감도 설정이 없으면 고정 시각의 그림자만 넣는다."""
    view = choose_view(collected, cfg) if "aerial_fov_degrees" in cfg else None
    _set_shadows(api, model, cfg, None if view is None else view.hour)
    if view is not None:
        _set_view(api, model, view)


def _set_shadows(api: Any, model: c_void_p, cfg: dict[str, Any], hour: float | None = None) -> None:
    """그림자를 켜고 위치와 시각을 넣는다. 실패해도 형상은 그대로라 멈추지 않는다."""
    info, value = c_void_p(), c_void_p()
    if "shadow_time_local" not in cfg or api.SUModelGetShadowInfo(model, byref(info)) or api.SUTypedValueCreate(byref(value)):
        return
    # 시계 시각을 UTC인 것처럼 숫자로 바꾼다. 스케치업이 이 숫자를 그 지역 시각으로 읽기 때문이다.
    moment = datetime.fromisoformat(cfg["shadow_time_local"]).replace(tzinfo=timezone.utc)
    if hour is not None:
        moment = moment.replace(hour=int(hour), minute=0)
    settings = (
        (b"DisplayShadows", api.SUTypedValueSetBool, c_bool(True)),
        (b"DisplayOnGroundPlane", api.SUTypedValueSetBool, c_bool(True)),
        (b"DisplayOnAllFaces", api.SUTypedValueSetBool, c_bool(True)),
        (b"Latitude", api.SUTypedValueSetDouble, c_double(cfg["shadow_latitude"])),
        (b"Longitude", api.SUTypedValueSetDouble, c_double(cfg["shadow_longitude"])),
        (b"TZOffset", api.SUTypedValueSetDouble, c_double(cfg["shadow_timezone"])),
        (b"ShadowTime_time_t", api.SUTypedValueSetInt32, c_int32(int(moment.timestamp()))),
        (b"Light", api.SUTypedValueSetInt32, c_int32(cfg["shadow_light"])),
        (b"Dark", api.SUTypedValueSetInt32, c_int32(cfg["shadow_dark"])),
    )
    for key, setter, number in settings:
        if not setter(value, number):
            api.SUShadowInfoSetValue(info, c_char_p(key), value)
    api.SUTypedValueRelease(byref(value))


def read_shadows(path: Path, sdk: Path) -> dict[str, float]:
    """저장한 파일의 그림자 설정을 다시 읽는다."""
    api = _load(sdk)
    model, info, value = c_void_p(), c_void_p(), c_void_p()
    _ok(api.SUModelCreateFromFile(byref(model), str(path).encode("utf-8")), "open")
    result: dict[str, float] = {}
    try:
        _ok(api.SUModelGetShadowInfo(model, byref(info)), "shadow info")
        _ok(api.SUTypedValueCreate(byref(value)), "value")
        flag, number = c_bool(), c_double()
        _ok(api.SUShadowInfoGetValue(info, b"DisplayShadows", byref(value)), "display")
        _ok(api.SUTypedValueGetBool(value, byref(flag)), "display value")
        result["DisplayShadows"] = float(flag.value)
        # 설정값이 아니라 SDK가 계산한 해 방향이다. 0보다 작으면 밤이라 화면 전체가 어두워진다.
        sun = (c_double * 3)()
        _ok(api.SUShadowInfoGetValue(info, b"SunDirection", byref(value)), "sun")
        _ok(api.SUTypedValueGetVector3d(value, sun), "sun value")
        result["SunX"], result["SunY"], result["SunZ"] = sun[0], sun[1], sun[2]
        for key in (b"Latitude", b"Longitude", b"TZOffset"):
            _ok(api.SUShadowInfoGetValue(info, key, byref(value)), "shadow value")
            _ok(api.SUTypedValueGetDouble(value, byref(number)), "shadow number")
            result[key.decode()] = number.value
        api.SUTypedValueRelease(byref(value))
    finally:
        api.SUModelRelease(byref(model))
    return result


def _fill(
    api: Any, entities: c_void_p, mesh: Mesh, triangles: bool = False,
    layer: c_void_p | None = None, material: c_void_p | None = None,
) -> None:
    geometry = c_void_p()
    _ok(api.SUGeometryInputCreate(byref(geometry)), "geometry")
    try:
        points = (_Point * len(mesh.vertices))(*[
            _Point(x / _MM_PER_INCH, y / _MM_PER_INCH, z / _MM_PER_INCH) for x, y, z in mesh.vertices.tolist()
        ])
        _ok(api.SUGeometryInputSetVertices(geometry, c_size_t(len(mesh.vertices)), points), "vertices")

        def loop_of(ring: list[int]) -> c_void_p:
            loop = c_void_p()
            _ok(api.SULoopInputCreate(byref(loop)), "loop")
            for vertex in ring:
                _ok(api.SULoopInputAddVertexIndex(loop, c_size_t(vertex)), "loop vertex")
            if layer is not None:
                for edge in range(len(ring)):
                    _ok(api.SULoopInputEdgeSetLayer(loop, c_size_t(edge), layer), "edge layer")
            for edge, start in enumerate(ring):
                end = ring[(edge + 1) % len(ring)]
                if (min(start, end), max(start, end)) in mesh.soft:
                    _ok(api.SULoopInputEdgeSetSoft(loop, c_size_t(edge), c_bool(True)), "edge soft")
                    _ok(api.SULoopInputEdgeSetSmooth(loop, c_size_t(edge), c_bool(True)), "edge smooth")
            return loop

        faces = mesh.polygons if not triangles else [(list(t), []) for t in mesh.triangles.tolist()]
        for outer, inners in faces:
            face = c_size_t()
            loop = loop_of(outer)
            _ok(api.SUGeometryInputAddFace(geometry, byref(loop), byref(face)), "face")
            if layer is not None:
                _ok(api.SUGeometryInputFaceSetLayer(geometry, face, layer), "face layer")
            if material is not None:
                front = _MaterialInput(num_uv_coords=0, material=material)
                _ok(api.SUGeometryInputFaceSetFrontMaterial(geometry, face, byref(front)), "face material")
            for ring in inners:
                inner = loop_of(ring)
                _ok(api.SUGeometryInputFaceAddInnerLoop(geometry, face, byref(inner)), "inner loop")
        _ok(api.SUEntitiesFill(entities, geometry, c_bool(True)), "fill")
    finally:
        api.SUGeometryInputRelease(byref(geometry))


def write_skp(
    storeys: list[Any], path: Path, cfg: dict[str, Any], sdk: Path, legacy: tuple[Path, int] | None = None
) -> tuple[dict[str, int], dict[str, float]]:
    """(종류별 개수, 그룹 이름별 기대 부피 mm3)를 돌려준다. legacy는 함께 쓸 옛 형식 (경로, 버전)이다."""
    api = _load(sdk)
    collected, counts = _collect(storeys, cfg)
    model = c_void_p()
    _ok(api.SUModelCreate(byref(model)), "model")
    expected: dict[str, float] = {}
    groups_written = failed = retried = 0
    try:
        _set_units(api, model)
        root = c_void_p()
        _ok(api.SUModelGetEntities(model, byref(root)), "entities")
        materials: dict[str, c_void_p] = {}
        for name, (colour, alpha) in MATERIALS.items():
            material = c_void_p()
            _ok(api.SUMaterialCreate(byref(material)), "material")
            _ok(api.SUMaterialSetName(material, name.encode()), "material name")
            value = _Color(*[round(v * 255) for v in colour], 255)
            _ok(api.SUMaterialSetColor(material, byref(value)), "material color")
            if alpha < 1.0:
                _ok(api.SUMaterialSetOpacity(material, c_double(alpha)), "opacity")
                _ok(api.SUMaterialSetUseOpacity(material, c_bool(True)), "use opacity")
            _ok(api.SUModelAddMaterials(model, c_size_t(1), byref(material)), "add material")
            materials[name] = material

        for key, groups in collected:
            folder = c_void_p()
            _ok(api.SULayerFolderCreate(byref(folder), key.encode()), "folder")
            _ok(api.SUModelAddLayerFolder(model, folder), "add folder")
            for category, items in groups.items():
                layer = c_void_p()
                _ok(api.SULayerCreate(byref(layer)), "layer")
                _ok(api.SULayerSetName(layer, f"{key}_{_GROUPS[category]}".encode()), "layer name")
                _ok(api.SUModelAddLayers(model, c_size_t(1), byref(layer)), "add layer")
                _ok(api.SULayerFolderAddLayer(folder, layer), "folder layer")
                for name, mesh, volume in items:
                    identifier = f"{key}_{name}"
                    group, inside = c_void_p(), c_void_p()
                    _ok(api.SUGroupCreate(byref(group)), "group")
                    _ok(api.SUEntitiesAddGroup(root, group), "add group")
                    _ok(api.SUGroupGetEntities(group, byref(inside)), "group entities")
                    try:
                        _fill(api, inside, mesh)
                    except SketchUpError:
                        # 한 점에서 고리가 맞닿는 등 면으로 못 만드는 다각형이 있다. 삼각형으로 다시 채운다.
                        retried += 1
                        try:
                            _fill(api, inside, mesh, triangles=True)
                        except SketchUpError:
                            failed += 1
                            element = api.SUGroupToDrawingElement(group)
                            _ok(api.SUDrawingElementSetLayer(c_void_p(element), layer), "group layer")
                            _ok(api.SUGroupSetName(group, f"{identifier}_failed".encode()), "group name")
                            continue
                    _ok(api.SUGroupSetName(group, identifier.encode()), "group name")
                    element = api.SUGroupToDrawingElement(group)
                    _ok(api.SUDrawingElementSetLayer(c_void_p(element), layer), "group layer")
                    _ok(api.SUDrawingElementSetMaterial(c_void_p(element), materials[category]), "group material")
                    groups_written += 1
                    if volume is not None:
                        expected[identifier] = volume
        _prepare_view(api, model, collected, cfg)
        _ok(api.SUModelSaveToFile(model, str(path).encode("utf-8")), "save")
        if legacy is not None:
            _ok(api.SUModelSaveToFileWithVersion(model, str(legacy[0]).encode("utf-8"), c_int32(legacy[1])), "save legacy")
    finally:
        api.SUModelRelease(byref(model))
    for name in ("walls", "curtain_walls", "columns", "slabs", "roofs", "stairs", "rails", "doors", "windows", "openings", "skipped_opening"):
        counts.setdefault(name, 0)
    return {**counts, "groups": groups_written, "fill_retried": retried, "fill_failed": failed}, expected


def write_editable_skp(
    collected: list[tuple[str, dict[str, list[Any]]]], tags: dict[str, dict[str, str]], path: Path, legacy: Path | None, sdk: Path,
    hidden: tuple[str, ...] = (), cfg: dict[str, Any] | None = None,
) -> dict[str, int]:
    """파일 하나에 층마다 그룹 하나를 둔다. 그룹 안은 면과 선이라 더블클릭해 바로 고칠 수 있다.

    요소별 그룹은 만들지 않는다. 태그는 층_종류이고, 확신이 낮은 요소는 층_REVIEW_종류다.
    """
    api = _load(sdk)
    model, root = c_void_p(), c_void_p()
    _ok(api.SUModelCreate(byref(model)), "model")
    result = {"faces_written": 0, "retried": 0, "failed": 0}
    try:
        _set_units(api, model)
        _ok(api.SUModelGetEntities(model, byref(root)), "entities")
        materials: dict[str, c_void_p] = {}
        for name, (colour, alpha) in MATERIALS.items():
            material = c_void_p()
            _ok(api.SUMaterialCreate(byref(material)), "material")
            _ok(api.SUMaterialSetName(material, name.encode()), "material name")
            value = _Color(*[round(v * 255) for v in colour], 255)
            _ok(api.SUMaterialSetColor(material, byref(value)), "material color")
            if alpha < 1.0:
                _ok(api.SUMaterialSetOpacity(material, c_double(alpha)), "opacity")
                _ok(api.SUMaterialSetUseOpacity(material, c_bool(True)), "use opacity")
            _ok(api.SUModelAddMaterials(model, c_size_t(1), byref(material)), "add material")
            materials[name] = material
        for key, groups in collected:
            folder, group, inside = c_void_p(), c_void_p(), c_void_p()
            _ok(api.SULayerFolderCreate(byref(folder), key.encode()), "folder")
            _ok(api.SUModelAddLayerFolder(model, folder), "add folder")
            _ok(api.SUGroupCreate(byref(group)), "group")
            _ok(api.SUEntitiesAddGroup(root, group), "add group")
            _ok(api.SUGroupSetName(group, key.encode()), "group name")
            _ok(api.SUGroupGetEntities(group, byref(inside)), "group entities")
            layers: dict[str, c_void_p] = {}
            for category, items in groups.items():
                for name, mesh, _ in items:
                    flag = tags.get(key, {}).get(name.split("_")[0], "")
                    tag = f"{key}_{flag}{_GROUPS[category]}"
                    if tag not in layers:
                        layer = c_void_p()
                        _ok(api.SULayerCreate(byref(layer)), "layer")
                        _ok(api.SULayerSetName(layer, tag.encode()), "layer name")
                        _ok(api.SUModelAddLayers(model, c_size_t(1), byref(layer)), "add layer")
                        _ok(api.SULayerFolderAddLayer(folder, layer), "folder layer")
                        if flag in hidden:
                            _ok(api.SULayerSetVisibility(layer, c_bool(False)), "layer visibility")
                        layers[tag] = layer
                    try:
                        _fill(api, inside, mesh, layer=layers[tag], material=materials[category])
                        result["faces_written"] += len(mesh.polygons)
                    except SketchUpError:
                        result["retried"] += 1
                        try:
                            _fill(api, inside, mesh, triangles=True, layer=layers[tag], material=materials[category])
                            result["faces_written"] += len(mesh.triangles)
                        except SketchUpError:
                            result["failed"] += 1
        if cfg is not None:
            _prepare_view(api, model, collected, cfg)
        _ok(api.SUModelSaveToFile(model, str(path).encode("utf-8")), "save")
        if legacy is not None:
            _ok(api.SUModelSaveToFileWithVersion(model, str(legacy).encode("utf-8"), c_int32(SU5_2005)), "save legacy")
    finally:
        api.SUModelRelease(byref(model))
    return result


def read_editable_skp(path: Path, sdk: Path) -> dict[str, Any]:
    """다시 열어 층 그룹, 그룹 안 면 수와 중첩 그룹 수, 태그를 돌려준다."""
    api = _load(sdk)
    model, root = c_void_p(), c_void_p()
    _ok(api.SUModelCreateFromFile(byref(model), str(path).encode("utf-8")), "open")
    try:
        _ok(api.SUModelGetEntities(model, byref(root)), "entities")
        count = c_size_t()
        _ok(api.SUEntitiesGetNumGroups(root, byref(count)), "groups")
        handles = (c_void_p * count.value)()
        _ok(api.SUEntitiesGetGroups(root, count, handles, byref(got := c_size_t())), "groups")
        floors: dict[str, dict[str, int]] = {}
        for index in range(got.value):
            group = c_void_p(handles[index])
            inside = c_void_p()
            _ok(api.SUGroupGetEntities(group, byref(inside)), "inside")
            faces, nested = c_size_t(), c_size_t()
            _ok(api.SUEntitiesGetNumFaces(inside, byref(faces)), "faces")
            _ok(api.SUEntitiesGetNumGroups(inside, byref(nested)), "nested")
            floors[_text(api, api.SUGroupGetName, group)] = {"faces": faces.value, "nested": nested.value}
        loose = c_size_t()
        _ok(api.SUEntitiesGetNumFaces(root, byref(loose)), "root faces")
        layer_count = c_size_t()
        _ok(api.SUModelGetNumLayers(model, byref(layer_count)), "layers")
        layers = (c_void_p * layer_count.value)()
        _ok(api.SUModelGetLayers(model, layer_count, layers, byref(got := c_size_t())), "layers")
        tags = [_text(api, api.SULayerGetName, c_void_p(layers[i])) for i in range(got.value)]
        return {"floors": floors, "root_faces": loose.value, "tags": tags}
    finally:
        api.SUModelRelease(byref(model))


def read_groups(path: Path, sdk: Path) -> dict[str, dict[str, float]]:
    """파일을 다시 열어 그룹별 태그, 면 수, 부피(mm3), 열린 모서리 수를 돌려준다."""
    api = _load(sdk)
    model, root = c_void_p(), c_void_p()
    _ok(api.SUModelCreateFromFile(byref(model), str(path).encode("utf-8")), "open")
    result: dict[str, dict[str, Any]] = {}
    try:
        _ok(api.SUModelGetEntities(model, byref(root)), "entities")
        count = c_size_t()
        _ok(api.SUEntitiesGetNumGroups(root, byref(count)), "group count")
        groups = (c_void_p * count.value)()
        got = c_size_t()
        _ok(api.SUEntitiesGetGroups(root, count, groups, byref(got)), "groups")
        for index in range(got.value):
            group = groups[index]
            name = _text(api, api.SUGroupGetName, c_void_p(group))
            element = api.SUGroupToDrawingElement(c_void_p(group))
            layer = c_void_p()
            _ok(api.SUDrawingElementGetLayer(c_void_p(element), byref(layer)), "layer")
            tag = _text(api, api.SULayerGetName, layer)
            inside = c_void_p()
            _ok(api.SUGroupGetEntities(c_void_p(group), byref(inside)), "inside")
            face_count = c_size_t()
            _ok(api.SUEntitiesGetNumFaces(inside, byref(face_count)), "faces")
            faces = (c_void_p * face_count.value)()
            _ok(api.SUEntitiesGetFaces(inside, face_count, faces, byref(got_faces := c_size_t())), "faces")
            volume = 0.0
            edges_seen: dict[int, int] = {}
            for f in range(got_faces.value):
                area, plane = c_double(), _Plane()
                face = c_void_p(faces[f])
                _ok(api.SUFaceGetArea(face, byref(area)), "area")
                _ok(api.SUFaceGetPlane(face, byref(plane)), "plane")
                volume += area.value * (-plane.d) / 3.0
                edge_count = c_size_t()
                _ok(api.SUFaceGetNumEdges(face, byref(edge_count)), "edge count")
                edges = (c_void_p * edge_count.value)()
                _ok(api.SUFaceGetEdges(face, edge_count, edges, byref(got_edges := c_size_t())), "edges")
                for e in range(got_edges.value):
                    if edges[e] not in edges_seen:
                        number = c_size_t()
                        _ok(api.SUEdgeGetNumFaces(c_void_p(edges[e]), byref(number)), "edge faces")
                        edges_seen[edges[e]] = number.value
            result[name] = {
                "tag": tag, "faces": got_faces.value, "volume_mm3": volume * _MM_PER_INCH ** 3,
                # 면이 하나뿐인 모서리만 구멍이다. 계단 단처럼 맞닿은 모서리는 면을 넷 공유한다.
                "open_edges": sum(n < 2 for n in edges_seen.values()),
            }
    finally:
        api.SUModelRelease(byref(model))
    return result


def verify_skp(path: Path, expected: dict[str, float], sdk: Path) -> dict[str, int]:
    """공식 SDK로 다시 열어 그룹, 태그, 닫힌 형상, 부피를 개수로 돌려준다."""
    groups = read_groups(path, sdk)
    return {
        "groups": len(groups),
        "wrong_tag": sum(not g["tag"].startswith(name.split("_")[0] + "_") for name, g in groups.items()),
        "no_faces": sum(g["faces"] == 0 and not name.endswith("_failed") for name, g in groups.items()),
        "failed": sum(name.endswith("_failed") for name in groups),
        "not_closed": sum(g["open_edges"] > 0 for g in groups.values()),
        "volume_checked": len(expected),
        "volume_mismatch": sum(
            abs(groups[name]["volume_mm3"] - volume) > volume * _VOLUME_TOLERANCE
            for name, volume in expected.items() if name in groups
        ),
        "missing": sum(name not in groups for name in expected),
    }
