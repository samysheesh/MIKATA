"""Lecture directe des volumes NTFS (phase 12d) : MFT et journal de modifications USN.

C'est la méthode de référence (celle d'Everything / WizTree) : la MFT est lue via
`FSCTL_ENUM_USN_DATA` (des millions de fichiers en quelques secondes, sans parcourir le
disque), puis le journal `FSCTL_READ_USN_JOURNAL` rejoue les modifications (créations,
renommages, suppressions) — jamais de rescan complet.

Les structures et les ioctl sont Windows (pywin32, import paresseux) ; la reconstruction
des chemins et l'application des événements sont des fonctions pures, testables partout.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# FRN de la racine d'un volume NTFS (constante du système de fichiers).
ROOT_FRN = 0x0005000000000005
FILE_ATTRIBUTE_DIRECTORY = 0x10

# Raisons du journal USN qui comptent pour un index de noms.
USN_REASON_FILE_CREATE = 0x00000100
USN_REASON_FILE_DELETE = 0x00000200
USN_REASON_RENAME_OLD_NAME = 0x00001000
USN_REASON_RENAME_NEW_NAME = 0x00002000
ALL_REASONS = 0xFFFFFFFF

FSCTL_ENUM_USN_DATA = 0x000900B3
FSCTL_READ_USN_JOURNAL = 0x000900BB
FSCTL_QUERY_USN_JOURNAL = 0x000900F4
ERROR_HANDLE_EOF = 38

# Sous-arbres système sans intérêt pour l'utilisateur : jamais montrés.
_HIDDEN_TOP = {"$extend", "system volume information"}

_BUFFER_SIZE = 1024 * 1024  # 1 Mo par appel ioctl
_NTFS_EPOCH_S = 11_644_473_600  # 1601 -> 1970


class UsnRecordV2(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint),
        ("major", ctypes.c_ushort),
        ("minor", ctypes.c_ushort),
        ("file_reference_number", ctypes.c_ulonglong),
        ("parent_file_reference_number", ctypes.c_ulonglong),
        ("usn", ctypes.c_longlong),
        ("timestamp", ctypes.c_longlong),
        ("reason", ctypes.c_uint),
        ("source_info", ctypes.c_uint),
        ("security_id", ctypes.c_uint),
        ("file_attributes", ctypes.c_uint),
        ("file_name_length", ctypes.c_ushort),
        ("file_name_offset", ctypes.c_ushort),
        ("file_name", ctypes.c_wchar * 1),
    ]


class MftEnumDataV0(ctypes.Structure):
    _fields_ = [
        ("start_file_reference_number", ctypes.c_ulonglong),
        ("low_usn", ctypes.c_longlong),
        ("high_usn", ctypes.c_longlong),
    ]


class ReadUsnJournalDataV0(ctypes.Structure):
    _fields_ = [
        ("start_usn", ctypes.c_longlong),
        ("reason_mask", ctypes.c_uint),
        ("return_only_on_close", ctypes.c_uint),
        ("timeout", ctypes.c_ulonglong),
        ("bytes_to_wait_for", ctypes.c_ulonglong),
        ("usn_journal_id", ctypes.c_ulonglong),
    ]


class UsnJournalDataV0(ctypes.Structure):
    _fields_ = [
        ("usn_journal_id", ctypes.c_ulonglong),
        ("first_usn", ctypes.c_longlong),
        ("next_usn", ctypes.c_longlong),
        ("lowest_valid_usn", ctypes.c_longlong),
        ("max_usn", ctypes.c_longlong),
        ("maximum_size", ctypes.c_ulonglong),
        ("allocation_delta", ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class UsnEntry:
    """Une entrée de la MFT (index initial)."""

    frn: int
    parent_frn: int
    name: str
    mtime: float  # secondes depuis 1970
    is_dir: bool


@dataclass(frozen=True)
class UsnEvent:
    """Un événement du journal (mise à jour incrémentale)."""

    frn: int
    parent_frn: int
    name: str
    reason: int
    mtime: float
    usn: int
    is_dir: bool


def _bytes_of(struct: ctypes.Structure) -> bytes:
    return ctypes.string_at(ctypes.byref(struct), ctypes.sizeof(struct))


def _timestamp_to_epoch(value: int) -> float:
    return value / 10_000_000 - _NTFS_EPOCH_S


def _parse_records(buffer: bytes) -> Iterator[tuple[UsnRecordV2, str]]:
    """Découpe un tampon ioctl en enregistrements USN (nom décodé UTF-16)."""
    offset = 0
    header = ctypes.sizeof(UsnRecordV2)
    while offset + header <= len(buffer):
        record = UsnRecordV2.from_buffer_copy(buffer, offset)
        if record.record_length < header or offset + record.record_length > len(buffer):
            break
        raw = buffer[
            offset + record.file_name_offset : offset
            + record.file_name_offset
            + record.file_name_length
        ]
        try:
            name = raw.decode("utf-16-le", errors="replace")
        except ValueError:
            name = ""
        yield record, name
        offset += record.record_length


def list_ntfs_volumes() -> list[str]:
    """Lettres des lecteurs NTFS (« C:\\ »...). Hors Windows : liste vide."""
    import sys

    if sys.platform != "win32":
        return []
    import ctypes as _ctypes

    kernel32 = _ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetVolumeInformationW.argtypes = [
        _ctypes.c_wchar_p,
        _ctypes.c_wchar_p,
        _ctypes.c_uint,
        _ctypes.POINTER(_ctypes.c_ulong),
        _ctypes.POINTER(_ctypes.c_ulong),
        _ctypes.POINTER(_ctypes.c_ulong),
        _ctypes.c_wchar_p,
        _ctypes.c_uint,
    ]
    kernel32.GetVolumeInformationW.restype = _ctypes.c_int
    drives = kernel32.GetLogicalDrives()
    volumes: list[str] = []
    for index in range(26):
        if not drives & (1 << index):
            continue
        letter = f"{chr(65 + index)}:\\"
        fs_buffer = _ctypes.create_unicode_buffer(64)
        serial = _ctypes.c_ulong()
        max_component = _ctypes.c_ulong()
        flags = _ctypes.c_ulong()
        ok = kernel32.GetVolumeInformationW(
            letter,
            None,
            0,
            _ctypes.byref(serial),
            _ctypes.byref(max_component),
            _ctypes.byref(flags),
            fs_buffer,
            64,
        )
        if ok and fs_buffer.value.casefold() == "ntfs":
            volumes.append(letter)
    return volumes


def _open_volume(volume: str) -> int:
    import win32con
    import win32file

    return win32file.CreateFileW(
        "\\\\.\\" + volume.rstrip("\\"),
        win32con.GENERIC_READ,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
    )


def can_read_volumes() -> bool:
    """L'accès direct aux volumes (MFT + journal) est-il autorisé ? Administrateur."""
    volumes = list_ntfs_volumes()
    if not volumes:
        return False
    try:
        import win32file

        handle = _open_volume(volumes[0])
        win32file.CloseHandle(handle)
        return True
    except Exception:
        return False


def enumerate_files(volume: str) -> Iterator[UsnEntry]:
    """Toutes les entrées de la MFT du volume (FRN, parent, nom, date, dossier ?)."""
    import win32file

    handle = _open_volume(volume)
    request = MftEnumDataV0()
    request.start_file_reference_number = 0
    request.low_usn = 0
    request.high_usn = 0x7FFFFFFFFFFFFFFF
    input_buffer = _bytes_of(request)
    last_frn = 0
    try:
        while True:
            try:
                output = win32file.DeviceIoControl(
                    handle, FSCTL_ENUM_USN_DATA, input_buffer, _BUFFER_SIZE
                )
            except win32file.error as exc:
                if exc.winerror == ERROR_HANDLE_EOF:
                    break
                raise
            if not output:
                break
            produced = 0
            for record, name in _parse_records(output[8:]):  # 8 octets d'en-tête USN
                produced += 1
                last_frn = record.file_reference_number
                if not name or ":" in name:  # flux alternatifs : pas des fichiers visibles
                    continue
                yield UsnEntry(
                    frn=record.file_reference_number,
                    parent_frn=record.parent_file_reference_number,
                    name=name,
                    mtime=_timestamp_to_epoch(record.timestamp),
                    is_dir=bool(record.file_attributes & FILE_ATTRIBUTE_DIRECTORY),
                )
            if produced == 0:
                break  # MFT terminée (le tampon peut être partiel sans être la fin)
            request.start_file_reference_number = last_frn
            input_buffer = _bytes_of(request)
    finally:
        win32file.CloseHandle(handle)


def query_journal(volume: str) -> tuple[int, int]:
    """(identifiant du journal, prochain USN attendu) du volume."""
    import win32file

    handle = _open_volume(volume)
    try:
        output = win32file.DeviceIoControl(handle, FSCTL_QUERY_USN_JOURNAL, None, 512)
        info = UsnJournalDataV0.from_buffer_copy(output)
        return info.usn_journal_id, info.next_usn
    finally:
        win32file.CloseHandle(handle)


def _journal_io(handle: int, ioctl: int, input_bytes: bytes, size: int) -> bytes:
    """DeviceIoControl avec un tampon de sortie aligné sur 64 Ko (exigé par le journal
    USN : un buffer Python classique n'est pas aligné, d'où l'erreur 87). Les types
    ctypes sont déclarés : sans `restype`, le pointeur 64 bits de VirtualAlloc serait
    tronqué à 32 bits (erreur 1784, tampon invalide)."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong,
    ]
    kernel32.VirtualFree.restype = ctypes.c_int
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
    kernel32.DeviceIoControl.restype = ctypes.c_int
    kernel32.DeviceIoControl.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
    ]
    buffer = kernel32.VirtualAlloc(None, size, 0x3000, 0x04)  # MEM_COMMIT|RESERVE, RW
    if not buffer:
        raise OSError("VirtualAlloc a échoué")
    try:
        returned = ctypes.c_ulong()
        ok = kernel32.DeviceIoControl(
            ctypes.c_void_p(int(handle)), ioctl,
            input_bytes, len(input_bytes),
            buffer, size,
            ctypes.byref(returned), None,
        )
        if not ok:
            raise ctypes.WinError()
        return ctypes.string_at(buffer, returned.value)
    finally:
        kernel32.VirtualFree(buffer, 0, 0x8000)  # MEM_RELEASE


def read_journal(
    volume: str, start_usn: int, journal_id: int, timeout_s: float = 0.0
) -> tuple[list[UsnEvent], int]:
    """Événements du journal depuis `start_usn` ; renvoie aussi le prochain USN à lire.

    `timeout_s = 0` : non bloquant (le poller reviendra au tour suivant). Lève sur une
    erreur réelle (journal vidé, volume disparu) : l'appelant reconstruit l'index.
    """
    import win32file

    handle = _open_volume(volume)
    request = ReadUsnJournalDataV0()
    request.start_usn = start_usn
    request.reason_mask = ALL_REASONS
    request.return_only_on_close = 0
    request.timeout = int(timeout_s)
    request.bytes_to_wait_for = 1
    request.usn_journal_id = journal_id
    events: list[UsnEvent] = []
    next_usn = start_usn
    try:
        while True:
            try:
                output = _journal_io(
                    handle, FSCTL_READ_USN_JOURNAL, _bytes_of(request), _BUFFER_SIZE
                )
            except OSError as exc:
                if exc.winerror == ERROR_HANDLE_EOF:
                    break  # rien de nouveau
                raise
            if len(output) <= 8:
                break
            produced = 0
            for record, name in _parse_records(output[8:]):  # 8 octets d'en-tête USN
                produced += 1
                next_usn = record.usn + 1
                if not name or ":" in name:
                    continue
                events.append(
                    UsnEvent(
                        frn=record.file_reference_number,
                        parent_frn=record.parent_file_reference_number,
                        name=name,
                        reason=record.reason,
                        mtime=_timestamp_to_epoch(record.timestamp),
                        usn=record.usn,
                        is_dir=bool(record.file_attributes & FILE_ATTRIBUTE_DIRECTORY),
                    )
                )
            if produced == 0:
                break
            request.start_usn = next_usn
    finally:
        win32file.CloseHandle(handle)
    return events, next_usn


def diagnose(volume: str) -> str:
    """Décodage du premier tampon de FSCTL_ENUM_USN_DATA (diagnostic, volume élevé)."""
    import win32file

    handle = _open_volume(volume)
    request = MftEnumDataV0()
    request.start_file_reference_number = 0
    request.low_usn = 0
    request.high_usn = 0x7FFFFFFFFFFFFFFF
    try:
        output = win32file.DeviceIoControl(
            handle, FSCTL_ENUM_USN_DATA, _bytes_of(request), _BUFFER_SIZE
        )
        parts = [f"{len(output)} octets"]
        if len(output) > 72:
            record = UsnRecordV2.from_buffer_copy(output, 8)
            raw_name = output[
                8 + record.file_name_offset : 8 + record.file_name_offset
                + record.file_name_length
            ]
            try:
                name = raw_name.decode("utf-16-le", errors="replace")
            except ValueError:
                name = "<decode impossible>"
            parts.append(
                f"record@8 len={record.record_length} "
                f"frn={record.file_reference_number:#x} "
                f"parent={record.parent_file_reference_number:#x} "
                f"name_len={record.file_name_length} name_off={record.file_name_offset} "
                f"name={name!r}"
            )
        parsed = sum(1 for _ in _parse_records(output[8:]))
        names: list[str] = []
        for _, name in _parse_records(output[8:]):
            names.append(repr(name))
            if len(names) >= 5:
                break
        parts.append(f"parsed={parsed} noms={', '.join(names)}")
        return " | ".join(parts)
    except win32file.error as exc:
        return f"erreur {exc.winerror} : {exc}"
    finally:
        win32file.CloseHandle(handle)


# -- parties pures (testables sur n'importe quelle plateforme) -------------------------


def build_paths(entries: Iterable[UsnEntry], volume: str) -> Iterator[tuple[str, UsnEntry]]:
    """Chemins complets reconstruits depuis la carte FRN -> (parent, nom).

    Les FRN dont le parent est inconnu (fichier supprimé pendant la lecture) sont sautés.
    Les sous-arbres système ($Extend, System Volume Information) ne sont pas produits.
    """
    volume = volume.rstrip("\\")
    by_frn: dict[int, UsnEntry] = {entry.frn: entry for entry in entries}
    resolved: dict[int, str | None] = {}

    def resolve(frn: int) -> str | None:
        if frn == ROOT_FRN:
            return volume + "\\"
        if frn in resolved:
            return resolved[frn]
        resolved[frn] = None  # cycle de sécurité
        entry = by_frn.get(frn)
        if entry is None:
            return None
        parent = resolve(entry.parent_frn)
        if parent is None:
            return None
        path = f"{parent}{entry.name}" + ("\\" if entry.is_dir else "")
        resolved[frn] = path
        return path

    for frn, entry in by_frn.items():
        if entry.is_dir:
            continue  # le chemin d'un dossier se résout par ses parents ; on l'émettra
        path = resolve(frn)
        if path is None:
            continue
        top = path[len(volume) + 1 :].split("\\", 1)[0].casefold()
        if top in _HIDDEN_TOP:
            continue
        yield path, entry

    for frn, entry in by_frn.items():
        if not entry.is_dir:
            continue
        path = resolve(frn)
        if path is None or path == f"{volume}\\":
            continue
        top = path[len(volume) + 1 :].split("\\", 1)[0].casefold()
        if top in _HIDDEN_TOP:
            continue
        yield path, entry


def relevant(reason: int) -> bool:
    """L'événement change-t-il un nom ou une existence dans l'index ?"""
    return bool(
        reason
        & (
            USN_REASON_FILE_CREATE
            | USN_REASON_FILE_DELETE
            | USN_REASON_RENAME_OLD_NAME
            | USN_REASON_RENAME_NEW_NAME
        )
    )
