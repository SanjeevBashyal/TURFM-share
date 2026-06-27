from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


DIRECT_RUN_SOURCE = "folder"
DIRECT_RUN_MASTER_PROJECT_PATH = Path(r"E:\TUR_FM\Nepal_Organize\3.Packages\20260415\4_Recieved Data\Cycle-1\Group-3copy")
DIRECT_RUN_PREPARE_FULL_STRUCTURE = False
BUR_INPUT_DIRNAME = "1 Bur"
LEGACY_BUR_INPUT_DIRNAME = "1 Bur-Bur"


@dataclass(frozen=True)
class FolderEntry:
    """One folder entry in the DTM-only project tree."""

    name: str
    level: int
    relative_parts: tuple[str, ...]
    contents: str = ""

    @property
    def relative_path(self) -> Path:
        return Path(*self.relative_parts)


@dataclass(frozen=True)
class SubProjectPaths:
    """Resolved paths for one DTM project/sub-project run."""

    project_name: str
    sub_project_name: str
    project_path: str
    sub_project_path: str
    hecras_project_path: str
    hecras_sub_project_path: str
    gis_project_path: str
    gis_sub_project_path: str
    dtm_project_path: str
    dtm_sub_project_path: str
    output_project_path: str
    output_sub_project_path: str
    temp_project_path: str
    temp_sub_project_path: str
    cross_section_path: str
    cross_section_file_path: str
    bank_line_path: str
    bank_line_file_path: str
    dtm_path: str


@dataclass
class Config:
    """Standalone DTM-only project config.

    This intentionally does not import or inherit from ``configProject.py``.
    It keeps only the folder layout needed by ``implementationDTM.py``:
    ``0 Essentials``, ``1 Bur``, ``2 GIS``, ``3 DTM``, and ``Z Temp``.
    """

    structure_source: str = "folder"
    master_project_path: str | None = None
    project_folder: str | None = None
    create_master_folder_if_missing: bool = False
    PROJECT_NAME: str = ""
    SUB_PROJECT_NAME: str = ""
    PROJECT_SHORT_NAME: str = ""

    BLEND_TYPE: str = "linear"
    HECRAS_VERSION: str = "6.6"
    DEM_FILENAME: str = "SET4_27_DTM_070226_R1.tif"
    DTM_INDEX_FILENAME: str = "dtm.csv"
    DTM_DIRNAME: str = "DTMs"
    CROSS_SECTION_DIRNAME: str = "KESIT_TESLIM"
    BANK_LINE_DIRNAME: str = "SEV_USTU"
    IGNORED_PROJECT_FOLDER_NAMES: tuple[str, ...] = ("Z Received",)

    MASTER_PATH: str = field(init=False)
    PROJECT_FOLDER: str = field(init=False)
    PROJECT_LONG_NAME: str = field(init=False, default="")
    PROJECT_GROUP: str = field(init=False, default="")
    FOLDER_ENTRIES: list[FolderEntry] = field(init=False, default_factory=list)
    FOLDER_DESCRIPTIONS: dict[str, str] = field(init=False, default_factory=dict)
    PATHS: dict[str, str] = field(init=False, default_factory=dict)

    ESSENTIALS_PATH: str = field(init=False)
    BUR_BUR_PATH: str = field(init=False)
    GIS_PATH: str = field(init=False)
    DTM_PATH: str = field(init=False)
    DTM_OUTPUT_PATH: str = field(init=False)
    TEMP_PATH: str = field(init=False)

    # Compatibility aliases used by old DTM helpers; these do not create extra
    # top-level folders in this DTM-only config.
    HEC_PATH: str = field(init=False)
    OUTPUT_PATH: str = field(init=False)
    OUTPUT_ROOT_PATH: str = field(init=False)

    PROJECT_DATA_PATH: str = field(init=False)
    HEC_PROJECT_PATH: str = field(init=False)
    HEC_SUB_PROJECT_PATH: str = field(init=False)
    GIS_PROJECT_PATH: str = field(init=False)
    GIS_SUB_PROJECT_PATH: str = field(init=False)
    DTM_PROJECT_PATH: str = field(init=False)
    DTM_SUB_PROJECT_PATH: str = field(init=False)
    OUTPUT_PROJECT_PATH: str = field(init=False)
    OUTPUT_SUB_PROJECT_PATH: str = field(init=False)
    TEMP_PROJECT_PATH: str = field(init=False)
    TEMP_SUB_PROJECT_PATH: str = field(init=False)

    DEM_PATH: str = field(init=False)
    DTM_INDEX_PATH: str = field(init=False)
    DTM_FOLDER_PATH: str = field(init=False)
    CROSS_SECTION_PATH: str = field(init=False)
    CROSS_SECTION_FILE_PATH: str = field(init=False)
    BANK_LINE_PATH: str = field(init=False)
    BANK_LINE_FILE_PATH: str = field(init=False)

    @classmethod
    def from_master_folder(
        cls,
        master_project_path: str,
        *,
        create_if_missing: bool = False,
        **kwargs,
    ) -> "Config":
        return cls(
            structure_source="folder",
            master_project_path=master_project_path,
            project_folder=master_project_path,
            create_master_folder_if_missing=create_if_missing,
            **kwargs,
        )

    def __post_init__(self):
        self.structure_source = self._normalize_structure_source(self.structure_source)
        explicit_master_path = self._clean_cell(self.master_project_path or self.project_folder or "")
        if not explicit_master_path:
            raise ValueError("master_project_path or project_folder must be provided.")
        self._initialize_from_master_folder(explicit_master_path)

    def _initialize_from_master_folder(self, folder_path: str):
        resolved_root = Path(self._clean_cell(folder_path)).expanduser()
        if not resolved_root.exists():
            if not self.create_master_folder_if_missing:
                raise FileNotFoundError(f"Master project folder does not exist: {resolved_root}")
            resolved_root.mkdir(parents=True, exist_ok=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(f"Master project path is not a directory: {resolved_root}")

        self.MASTER_PATH = str(resolved_root)
        self.PROJECT_FOLDER = str(resolved_root)
        self._build_folder_entries_from_master_folder(resolved_root)
        self._ensure_top_level_entries()
        self._rebuild_paths()
        self._refresh_compatibility_paths()

    def _build_folder_entries_from_master_folder(self, root: Path):
        self.FOLDER_ENTRIES = []
        added: set[tuple[str, ...]] = set()

        def add_entry(relative_parts: tuple[str, ...]):
            if not relative_parts or relative_parts in added:
                return
            added.add(relative_parts)
            self.FOLDER_ENTRIES.append(
                FolderEntry(
                    name=relative_parts[-1],
                    level=len(relative_parts) - 1,
                    relative_parts=relative_parts,
                )
            )

        for top_level_name in self._default_top_level_names():
            add_entry((top_level_name,))

        bur_bur_path = self._resolve_bur_input_root(root)
        if bur_bur_path.is_dir():
            for project_dir in sorted(
                [
                    path
                    for path in bur_bur_path.iterdir()
                    if self._looks_like_project_directory(path)
                ],
                key=lambda item: item.name.upper(),
            ):
                sub_project_dirs = [path for path in project_dir.iterdir() if path.is_dir()]
                if not sub_project_dirs:
                    continue
                add_entry((bur_bur_path.name, project_dir.name))
                for sub_project_dir in sorted(
                    sub_project_dirs,
                    key=lambda item: (self._version_number(item.name), item.name.upper()),
                    reverse=True,
                ):
                    add_entry((bur_bur_path.name, project_dir.name, sub_project_dir.name))

    def _ensure_top_level_entries(self):
        added = {entry.relative_parts for entry in self.FOLDER_ENTRIES}
        for top_level_name in self._default_top_level_names():
            relative_parts = (top_level_name,)
            if relative_parts in added:
                continue
            self.FOLDER_ENTRIES.append(
                FolderEntry(
                    name=top_level_name,
                    level=0,
                    relative_parts=relative_parts,
                )
            )

    def _rebuild_paths(self):
        self.PATHS = {"PROJECT_FOLDER": self.PROJECT_FOLDER}
        self.FOLDER_DESCRIPTIONS = {}
        for entry in self.FOLDER_ENTRIES:
            relative_key = entry.relative_path.as_posix()
            absolute_path = str(Path(self.PROJECT_FOLDER) / entry.relative_path)
            self.PATHS[relative_key] = absolute_path
            self.FOLDER_DESCRIPTIONS[relative_key] = entry.contents
            setattr(self, self._to_attr_name(entry.relative_parts), absolute_path)

    def _refresh_compatibility_paths(self):
        self._select_active_names_from_structure_if_missing()
        self.PROJECT_LONG_NAME = self._project_long_name_from_short_name(self.PROJECT_SHORT_NAME)

        self.ESSENTIALS_PATH = self._configured_path_or_default("0 Essentials")
        self.BUR_BUR_PATH = self._configured_path_or_default(
            self._bur_input_relative_key()
        )
        self.GIS_PATH = self._configured_path_or_default("2 GIS")
        self.DTM_OUTPUT_PATH = self._configured_path_or_default("3 DTM")
        self.DTM_PATH = self.DTM_OUTPUT_PATH
        self.TEMP_PATH = self._configured_path_or_default("Z Temp")

        self.HEC_PATH = self.DTM_OUTPUT_PATH
        self.OUTPUT_ROOT_PATH = self.GIS_PATH
        self.OUTPUT_PATH = self.GIS_PATH

        self.DTM_INDEX_PATH = str(Path(self.ESSENTIALS_PATH) / self.DTM_INDEX_FILENAME)
        self.DTM_FOLDER_PATH = str(Path(self.ESSENTIALS_PATH) / self.DTM_DIRNAME)

        project_relative_path = self._resolve_project_relative_path()
        self.PROJECT_GROUP = self._project_group_from_relative_path(project_relative_path)
        self.PROJECT_DATA_PATH = self._absolute_from_relative_path(project_relative_path)
        active_project_name = self.PROJECT_GROUP or self.PROJECT_NAME

        self.GIS_PROJECT_PATH = str(Path(self.GIS_PATH) / active_project_name)
        self.GIS_SUB_PROJECT_PATH = str(Path(self.GIS_PROJECT_PATH) / self.SUB_PROJECT_NAME)
        self.DTM_PROJECT_PATH = str(Path(self.DTM_OUTPUT_PATH) / active_project_name)
        self.DTM_SUB_PROJECT_PATH = str(Path(self.DTM_PROJECT_PATH) / self.SUB_PROJECT_NAME)
        self.TEMP_PROJECT_PATH = str(Path(self.TEMP_PATH) / active_project_name)
        self.TEMP_SUB_PROJECT_PATH = str(Path(self.TEMP_PROJECT_PATH) / self.SUB_PROJECT_NAME)

        self.HEC_PROJECT_PATH = self.DTM_PROJECT_PATH
        self.HEC_SUB_PROJECT_PATH = self.DTM_SUB_PROJECT_PATH
        self.OUTPUT_PROJECT_PATH = self.GIS_PROJECT_PATH
        self.OUTPUT_SUB_PROJECT_PATH = self.GIS_SUB_PROJECT_PATH

        self.DEM_PATH = self.resolve_dtm_path(
            project_name=active_project_name,
            sub_project_name=self.SUB_PROJECT_NAME,
            required=False,
        )
        self.CROSS_SECTION_PATH = str(Path(self.PROJECT_DATA_PATH) / self.CROSS_SECTION_DIRNAME)
        self.CROSS_SECTION_FILE_PATH = str(
            Path(self.CROSS_SECTION_PATH)
            / f"{self.PROJECT_LONG_NAME}_{self.CROSS_SECTION_DIRNAME}.csv"
        )
        self.BANK_LINE_PATH = str(Path(self.PROJECT_DATA_PATH) / self.BANK_LINE_DIRNAME)
        self.BANK_LINE_FILE_PATH = str(
            Path(self.BANK_LINE_PATH) / f"{self.PROJECT_LONG_NAME}_{self.BANK_LINE_DIRNAME}.shp"
        )

    def _select_active_names_from_structure_if_missing(self):
        project_name = self._clean_cell(self.PROJECT_NAME)
        sub_project_name = self._clean_cell(self.SUB_PROJECT_NAME)
        project_short_name = self._clean_cell(self.PROJECT_SHORT_NAME)

        project_subprojects: dict[str, list[str]] = {}
        for entry in self.FOLDER_ENTRIES:
            parts = entry.relative_parts
            if len(parts) == 2 and self._is_bur_input_key(parts[0]):
                project_subprojects.setdefault(parts[1], [])
            elif len(parts) == 3 and self._is_bur_input_key(parts[0]):
                project_subprojects.setdefault(parts[1], []).append(parts[2])

        if not project_name and project_subprojects:
            project_name = next(iter(project_subprojects))
        if not sub_project_name and project_name:
            sub_projects = project_subprojects.get(project_name, [])
            if sub_projects:
                sub_project_name = sub_projects[0]
        if not project_short_name:
            project_short_name = self._project_short_name_from_long_name(sub_project_name or project_name)

        self.PROJECT_NAME = project_name
        self.SUB_PROJECT_NAME = sub_project_name
        self.PROJECT_SHORT_NAME = project_short_name

    def get_sub_project_paths(
        self,
        project_name: str,
        sub_project_name: str,
        cross_section_stem: str | None = None,
        bank_line_stem: str | None = None,
        resolve_dtm: bool = False,
    ) -> SubProjectPaths:
        project_path = self.get_project_path(project_name)
        sub_project_path = self.get_sub_project_path(project_name, sub_project_name)
        cross_section_file_path = self.find_cross_section_file(
            project_name,
            sub_project_name,
            file_stem=cross_section_stem,
        )
        bank_line_file_path = self.find_bank_line_file(
            project_name,
            sub_project_name,
            file_stem=bank_line_stem,
        )
        dtm_path = self.resolve_dtm_path(project_name, sub_project_name) if resolve_dtm else ""
        gis_project_path = self.get_gis_project_path(project_name)
        dtm_project_path = self.get_dtm_project_path(project_name)
        temp_project_path = self.get_temp_project_path(project_name)
        gis_sub_project_path = str(Path(gis_project_path) / Path(sub_project_path).name)
        dtm_sub_project_path = str(Path(dtm_project_path) / Path(sub_project_path).name)
        temp_sub_project_path = str(Path(temp_project_path) / Path(sub_project_path).name)

        return SubProjectPaths(
            project_name=Path(project_path).name,
            sub_project_name=Path(sub_project_path).name,
            project_path=project_path,
            sub_project_path=sub_project_path,
            hecras_project_path=dtm_project_path,
            hecras_sub_project_path=dtm_sub_project_path,
            gis_project_path=gis_project_path,
            gis_sub_project_path=gis_sub_project_path,
            dtm_project_path=dtm_project_path,
            dtm_sub_project_path=dtm_sub_project_path,
            output_project_path=gis_project_path,
            output_sub_project_path=gis_sub_project_path,
            temp_project_path=temp_project_path,
            temp_sub_project_path=temp_sub_project_path,
            cross_section_path=str(Path(cross_section_file_path).parent),
            cross_section_file_path=cross_section_file_path,
            bank_line_path=str(Path(bank_line_file_path).parent),
            bank_line_file_path=bank_line_file_path,
            dtm_path=dtm_path,
        )

    def get_project_path(self, project_name: str) -> str:
        return str(self._resolve_child_directory(Path(self.BUR_BUR_PATH), project_name))

    def get_sub_project_path(self, project_name: str, sub_project_name: str) -> str:
        return str(self._resolve_child_directory(Path(self.get_project_path(project_name)), sub_project_name))

    def get_gis_project_path(self, project_name: str) -> str:
        return str(self._resolve_child_directory(Path(self.GIS_PATH), project_name, create_fallback=True))

    def get_gis_sub_project_path(self, project_name: str, sub_project_name: str) -> str:
        return str(Path(self.get_gis_project_path(project_name)) / Path(self.get_sub_project_path(project_name, sub_project_name)).name)

    def get_dtm_project_path(self, project_name: str) -> str:
        return str(self._resolve_child_directory(Path(self.DTM_OUTPUT_PATH), project_name, create_fallback=True))

    def get_dtm_sub_project_path(self, project_name: str, sub_project_name: str) -> str:
        return str(Path(self.get_dtm_project_path(project_name)) / Path(self.get_sub_project_path(project_name, sub_project_name)).name)

    def get_temp_project_path(self, project_name: str) -> str:
        return str(self._resolve_child_directory(Path(self.TEMP_PATH), project_name, create_fallback=True))

    def get_temp_sub_project_path(self, project_name: str, sub_project_name: str) -> str:
        return str(Path(self.get_temp_project_path(project_name)) / Path(self.get_sub_project_path(project_name, sub_project_name)).name)

    def find_cross_section_file(
        self,
        project_name: str,
        sub_project_name: str,
        file_stem: str | None = None,
    ) -> str:
        return str(
            self._resolve_versioned_file(
                folder=Path(self.get_sub_project_path(project_name, sub_project_name)),
                file_stem=file_stem,
                default_contains=self.CROSS_SECTION_DIRNAME,
                extension=".csv",
            )
        )

    def find_bank_line_file(
        self,
        project_name: str,
        sub_project_name: str,
        file_stem: str | None = None,
    ) -> str:
        return str(
            self._resolve_versioned_file(
                folder=Path(self.get_sub_project_path(project_name, sub_project_name)),
                file_stem=file_stem,
                default_contains=self.BANK_LINE_DIRNAME,
                extension=".shp",
            )
        )

    def resolve_dtm_path(
        self,
        project_name: str | None = None,
        sub_project_name: str | None = None,
        dtm_name: str | None = None,
        required: bool = True,
    ) -> str:
        selected_dtm_name = self._clean_cell(dtm_name or "")
        if not selected_dtm_name:
            selected_dtm_name = self._lookup_dtm_name(project_name, sub_project_name) or ""

        if selected_dtm_name:
            try:
                return str(self._resolve_dtm_file(selected_dtm_name))
            except FileNotFoundError:
                if required:
                    raise

        if required and Path(self.DTM_INDEX_PATH).exists() and not selected_dtm_name:
            raise FileNotFoundError(
                f"No matching DTM row was found in {Path(self.DTM_INDEX_PATH)} for "
                f"project {project_name!r}, sub-project {sub_project_name!r}."
            )
        return str(self._resolve_default_dtm_path(required=required))

    def _lookup_dtm_name(
        self,
        project_name: str | None,
        sub_project_name: str | None,
    ) -> str | None:
        dtm_index_path = Path(self.DTM_INDEX_PATH)
        if not dtm_index_path.exists():
            return None

        fieldnames, rows = self._read_delimited_dict_rows(dtm_index_path)
        if not rows or not fieldnames:
            return None

        dtm_column = self._find_dtm_index_column(
            fieldnames,
            preferred=("DTM", "DTMName", "DTMPrefix", "DEM", "DEMName", "Raster", "RasterName", "TIF", "TIFF"),
            contains=("DTM", "DEM", "RASTER", "TIF", "TIFF"),
        )
        if dtm_column is None:
            raise ValueError(
                f"Could not find a DTM name column in {dtm_index_path}. "
                "Use a column such as 'DTM' or 'DTM Name'."
            )

        project_column = self._find_dtm_index_column(
            fieldnames,
            preferred=("Project", "ProjectName", "ProjectShortName", "MainProject"),
            contains=("PROJECT",),
            reject_contains=("SUB",),
        )
        sub_project_column = self._find_dtm_index_column(
            fieldnames,
            preferred=("SubProject", "SubProjectName", "SubProjectFolder", "SubProjectPath", "Channel", "River", "RiverName"),
            contains=("SUBPROJECT", "CHANNEL", "RIVER"),
        )

        project_name = self._clean_cell(project_name or "")
        sub_project_name = self._clean_cell(sub_project_name or "")
        best_score = -1
        best_dtm_name = None

        for row in rows:
            score = 0
            has_filter = False
            if project_column:
                project_value = self._clean_cell(row.get(project_column, ""))
                if project_value:
                    has_filter = True
                    if not self._names_match(project_name, project_value):
                        continue
                    score += 2
            if sub_project_column:
                sub_project_value = self._clean_cell(row.get(sub_project_column, ""))
                if sub_project_value:
                    has_filter = True
                    if not self._names_match(sub_project_name, sub_project_value):
                        continue
                    score += 4
            if not has_filter:
                first_value = self._clean_cell(row.get(fieldnames[0], ""))
                if not (
                    self._names_match(project_name, first_value)
                    or self._names_match(sub_project_name, first_value)
                ):
                    continue
                score += 1

            dtm_value = self._clean_cell(row.get(dtm_column, ""))
            if not dtm_value:
                continue
            if score > best_score:
                best_score = score
                best_dtm_name = dtm_value

        return best_dtm_name

    def _resolve_dtm_file(self, dtm_name: str) -> Path:
        clean_name = self._clean_cell(dtm_name)
        candidate_path = Path(clean_name).expanduser()
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path

        search_roots = [Path(self.DTM_FOLDER_PATH), Path(self.ESSENTIALS_PATH)]
        if not candidate_path.is_absolute() and candidate_path.parent != Path("."):
            search_roots.insert(0, Path(self.ESSENTIALS_PATH) / candidate_path.parent)
            clean_name = candidate_path.name

        patterns = []
        if any(char in clean_name for char in "*?"):
            patterns.append(clean_name)
        elif Path(clean_name).suffix.lower() in {".tif", ".tiff"}:
            patterns.extend([clean_name, f"{Path(clean_name).stem}*.tif", f"{Path(clean_name).stem}*.tiff"])
        else:
            patterns.extend([f"{clean_name}*.tif", f"{clean_name}*.tiff"])

        candidates: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                candidates.extend(
                    path
                    for path in root.glob(pattern)
                    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
                )

        if candidates:
            return self._select_best_path(list(dict.fromkeys(candidates)))

        available = self._available_dtm_raster_names()
        available_message = f" Available rasters: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"Could not find DTM raster matching {dtm_name!r}. "
            f"Searched {Path(self.DTM_FOLDER_PATH)} and {Path(self.ESSENTIALS_PATH)} using '<DTM name>*.tif'."
            f"{available_message}"
        )

    def _available_dtm_raster_names(self) -> list[str]:
        names = []
        for root in (Path(self.DTM_FOLDER_PATH), Path(self.ESSENTIALS_PATH)):
            if not root.exists():
                continue
            for path in root.glob("*"):
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
                    names.append(path.name)
        return sorted(set(names))

    def _resolve_default_dtm_path(self, required: bool = True) -> Path:
        candidates = [
            Path(self.ESSENTIALS_PATH) / self.DEM_FILENAME,
            Path(self.DTM_FOLDER_PATH) / self.DEM_FILENAME,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if required:
            raise FileNotFoundError(
                f"No DTM could be resolved. Add {Path(self.DTM_INDEX_PATH)} "
                f"or place {self.DEM_FILENAME!r} in {Path(self.DTM_FOLDER_PATH)}."
            )
        return candidates[0]

    def discover_project_subprojects(self) -> dict[str, list[str]]:
        project_subprojects: dict[str, list[str]] = {}
        for entry in self.FOLDER_ENTRIES:
            parts = entry.relative_parts
            if len(parts) == 2 and self._is_bur_input_key(parts[0]):
                project_subprojects.setdefault(parts[1], [])
            elif len(parts) == 3 and self._is_bur_input_key(parts[0]):
                project_subprojects.setdefault(parts[1], []).append(parts[2])
        return {project: sub_projects for project, sub_projects in project_subprojects.items() if sub_projects}

    def get_essential_directories(self) -> list[Path]:
        directories = {Path(self.PROJECT_FOLDER)}
        for top_level_name in self._default_top_level_names():
            relative_key = (
                self._bur_input_relative_key()
                if top_level_name == BUR_INPUT_DIRNAME
                else top_level_name
            )
            directories.add(Path(self.PATHS.get(relative_key, self._absolute_from_relative_path(relative_key))))
        return sorted(directories, key=lambda item: (len(item.parts), str(item)))

    def setup_essential_directories(self):
        for directory in self.get_essential_directories():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Ensured essential directory exists: {directory}")

    def setup_directories(self):
        directories = {
            *self.get_essential_directories(),
            *[
                Path(self.PATHS[entry.relative_path.as_posix()])
                for entry in self.FOLDER_ENTRIES
                if len(entry.relative_parts) > 1
            ],
            Path(self.PROJECT_DATA_PATH),
            Path(self.GIS_PROJECT_PATH),
            Path(self.GIS_SUB_PROJECT_PATH),
            Path(self.DTM_PROJECT_PATH),
            Path(self.DTM_SUB_PROJECT_PATH),
            Path(self.TEMP_PROJECT_PATH),
            Path(self.TEMP_SUB_PROJECT_PATH),
            Path(self.CROSS_SECTION_PATH),
            Path(self.BANK_LINE_PATH),
        }
        for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Ensured directory exists: {directory}")

    @classmethod
    def _read_delimited_dict_rows(cls, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        text = path.read_text(encoding="utf-8-sig")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return [], []
        sample = "\n".join(lines[:20])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            first_line = lines[0]
            delimiter = "\t" if "\t" in first_line else ";" if ";" in first_line else "|" if "|" in first_line else ","
            dialect = csv.excel()
            dialect.delimiter = delimiter
        reader = csv.DictReader(lines, dialect=dialect)
        if not reader.fieldnames:
            return [], []
        clean_fieldnames = [cls._clean_cell(field or "") for field in reader.fieldnames]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row: dict[str, str] = {}
            for raw_field, clean_field in zip(reader.fieldnames, clean_fieldnames):
                if clean_field:
                    row[clean_field] = cls._clean_cell(raw_row.get(raw_field, ""))
            if any(value for value in row.values()):
                rows.append(row)
        return [field for field in clean_fieldnames if field], rows

    @classmethod
    def _find_dtm_index_column(
        cls,
        fieldnames: list[str],
        preferred: tuple[str, ...],
        contains: tuple[str, ...] = (),
        reject_contains: tuple[str, ...] = (),
    ) -> str | None:
        normalized_lookup = {cls._normalize_name(field): field for field in fieldnames}
        for candidate in preferred:
            match = normalized_lookup.get(cls._normalize_name(candidate))
            if match is not None:
                return match
        for field in fieldnames:
            normalized = cls._normalize_name(field)
            if reject_contains and any(token in normalized for token in reject_contains):
                continue
            if contains and any(token in normalized for token in contains):
                return field
        return None

    def _resolve_child_directory(
        self,
        parent: Path,
        name: str,
        create_fallback: bool = False,
    ) -> Path:
        clean_name = self._clean_cell(name)
        exact_path = parent / clean_name
        if exact_path.is_dir():
            return exact_path
        if "*" in clean_name or "?" in clean_name:
            candidates = [path for path in parent.glob(clean_name) if path.is_dir()]
        else:
            normalized_name = self._normalize_name(clean_name)
            candidates = [
                path
                for path in parent.iterdir()
                if path.is_dir()
                and (
                    self._normalize_name(path.name) == normalized_name
                    or self._normalize_name(path.name).startswith(normalized_name)
                )
            ] if parent.is_dir() else []
        if candidates:
            return self._select_best_path(candidates)
        if create_fallback:
            return exact_path
        raise FileNotFoundError(f"Could not find folder matching {clean_name!r} inside {parent}")

    def _resolve_versioned_file(
        self,
        folder: Path,
        file_stem: str | None,
        default_contains: str,
        extension: str,
    ) -> Path:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        clean_extension = extension if extension.startswith(".") else f".{extension}"
        clean_stem = self._clean_cell(file_stem or "")
        pattern = f"*{default_contains}*{clean_extension}"
        if clean_stem:
            pattern = clean_stem if any(char in clean_stem for char in "*?") else f"{clean_stem}*"
            if Path(pattern).suffix == "":
                pattern = f"{pattern}{clean_extension}"

        candidates = [
            path
            for path in folder.rglob(pattern)
            if path.is_file() and path.suffix.lower() == clean_extension.lower()
        ]
        normalized_stem = self._normalize_name(clean_stem)
        normalized_contains = self._normalize_name(default_contains)
        if not candidates:
            candidates = [
                path
                for path in folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() == clean_extension.lower()
                and (
                    (
                        not clean_stem
                        and normalized_contains in self._normalize_name(path.stem)
                    )
                    or (
                        clean_stem
                        and self._normalize_name(path.stem).startswith(normalized_stem)
                    )
                )
            ]
        if candidates:
            return self._select_best_path(candidates)

        expected = (
            f"a {clean_extension} file containing {default_contains!r}"
            if not clean_stem
            else f"a {clean_extension} file matching {clean_stem!r}"
        )
        raise FileNotFoundError(f"Could not find {expected} anywhere inside sub-project folder: {folder}")

    @staticmethod
    def _clean_cell(value: str) -> str:
        return str(value).strip().strip('"')

    @staticmethod
    def _is_bur_input_key(value: str) -> bool:
        return value in {BUR_INPUT_DIRNAME, LEGACY_BUR_INPUT_DIRNAME}

    def _resolve_bur_input_root(self, root: Path) -> Path:
        preferred = root / BUR_INPUT_DIRNAME
        if preferred.is_dir():
            return preferred
        legacy = root / LEGACY_BUR_INPUT_DIRNAME
        if legacy.is_dir():
            return legacy
        return preferred

    def _bur_input_relative_key(self) -> str:
        preferred = Path(self.PROJECT_FOLDER) / BUR_INPUT_DIRNAME
        legacy = Path(self.PROJECT_FOLDER) / LEGACY_BUR_INPUT_DIRNAME
        if not preferred.is_dir() and legacy.is_dir():
            return LEGACY_BUR_INPUT_DIRNAME
        return BUR_INPUT_DIRNAME

    @classmethod
    def _normalize_structure_source(cls, value: str | None) -> str:
        normalized = cls._clean_cell(value or "folder").lower()
        aliases = {
            "folder": "folder",
            "filesystem": "folder",
            "path": "folder",
            "auto": "folder",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported structure_source {value!r}. Only 'folder' is supported.")
        return aliases[normalized]

    @staticmethod
    def _default_top_level_names() -> tuple[str, ...]:
        return (
            "0 Essentials",
            BUR_INPUT_DIRNAME,
            "2 GIS",
            "3 DTM",
            "4 Hecras",
            "5 Outputs",
            "Z Temp",
        )

    @staticmethod
    def _to_attr_name(parts: tuple[str, ...]) -> str:
        tokens: list[str] = []
        for part in parts:
            cleaned = re.sub(r"^\d+\s*", "", part.strip())
            cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned).strip("_").upper()
            if cleaned:
                tokens.append(cleaned)
        return "_".join(tokens) + "_PATH"

    def _configured_path_or_default(self, relative_key: str) -> str:
        return self.PATHS.get(relative_key, self._absolute_from_relative_path(relative_key))

    def _absolute_from_relative_path(self, relative_key: str) -> str:
        return str(Path(self.PROJECT_FOLDER) / Path(relative_key))

    def _resolve_project_relative_path(self) -> str:
        project_candidates = [
            entry.relative_path.as_posix()
            for entry in self.FOLDER_ENTRIES
            if entry.relative_parts and self._is_bur_input_key(entry.relative_parts[0])
        ]
        for target_name in (
            self.PROJECT_NAME,
            self.PROJECT_SHORT_NAME,
            self.PROJECT_LONG_NAME,
            self._normalize_name(self.PROJECT_NAME),
            self._normalize_name(self.PROJECT_SHORT_NAME),
            self._normalize_name(self.PROJECT_LONG_NAME),
        ):
            if not target_name:
                continue
            for candidate in project_candidates:
                candidate_name = Path(candidate).name
                if candidate_name == target_name or self._normalize_name(candidate_name) == target_name:
                    return candidate
        if self.PROJECT_NAME:
            return Path(self._bur_input_relative_key(), self.PROJECT_NAME).as_posix()
        if project_candidates:
            return project_candidates[0]
        return Path(self._bur_input_relative_key(), self.PROJECT_LONG_NAME).as_posix()

    @staticmethod
    def _project_group_from_relative_path(relative_key: str) -> str:
        parts = Path(relative_key).parts
        if len(parts) >= 3:
            return parts[1]
        return ""

    def _looks_like_project_directory(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        ignored_names = {self._normalize_name(name) for name in self.IGNORED_PROJECT_FOLDER_NAMES}
        return self._normalize_name(path.name) not in ignored_names

    def _select_best_path(self, paths: list[Path]) -> Path:
        return sorted(paths, key=self._path_sort_key, reverse=True)[0]

    def _path_sort_key(self, path: Path) -> tuple[int, float, str]:
        try:
            modified_time = path.stat().st_mtime
        except OSError:
            modified_time = 0.0
        return (self._version_number(path.name), modified_time, path.name.upper())

    @staticmethod
    def _version_number(name: str) -> int:
        matches = re.findall(r"(?:^|[_\-\s])V(\d+)(?=$|[^0-9])", name, flags=re.IGNORECASE)
        return max((int(match) for match in matches), default=0)

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "", value).upper()

    @classmethod
    def _names_match(cls, left: str, right: str) -> bool:
        left_norm = cls._normalize_name(cls._clean_cell(left or ""))
        right_norm = cls._normalize_name(cls._clean_cell(right or ""))
        if not left_norm or not right_norm:
            return False
        return left_norm == right_norm or left_norm.endswith(right_norm) or right_norm.endswith(left_norm)

    @staticmethod
    def _project_short_name_from_long_name(project_long_name: str) -> str:
        prefix = "BUR-BUR-MER-"
        if project_long_name.upper().startswith(prefix):
            return project_long_name[len(prefix):]
        return project_long_name

    @staticmethod
    def _project_long_name_from_short_name(project_short_name: str) -> str:
        project_short_name = str(project_short_name or "").strip()
        if not project_short_name:
            return ""
        prefix = "BUR-BUR-MER-"
        if project_short_name.upper().startswith(prefix):
            return project_short_name
        return f"{prefix}{project_short_name}"


def parse_direct_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the DTM-only master project folder layout."
    )
    parser.add_argument(
        "master_project_path",
        nargs="?",
        help=(
            "Master project folder path. If omitted, DIRECT_RUN_MASTER_PROJECT_PATH "
            "from configProjectDTM.py is used."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["folder"],
        help="Structure source to use. Only folder mode is supported.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Create project/sub-project DTM working folders as well.",
    )
    return parser.parse_args()


def build_direct_run_config(args: argparse.Namespace) -> Config:
    master_project_path = args.master_project_path or DIRECT_RUN_MASTER_PROJECT_PATH
    source = args.source or DIRECT_RUN_SOURCE or "folder"
    if not master_project_path:
        raise ValueError(
            "A master project folder path is required. Set DIRECT_RUN_MASTER_PROJECT_PATH "
            "or pass the folder path when running configProjectDTM.py."
        )
    return Config(
        structure_source=source,
        master_project_path=master_project_path,
        project_folder=master_project_path,
        create_master_folder_if_missing=True,
    )


if __name__ == "__main__":
    args = parse_direct_run_args()
    config = build_direct_run_config(args)
    print(f"Loaded DTM project structure for: {config.PROJECT_FOLDER}")
    if args.full or DIRECT_RUN_PREPARE_FULL_STRUCTURE:
        config.setup_directories()
    else:
        config.setup_essential_directories()
