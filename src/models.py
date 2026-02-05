"""Data models for EDL entries, source entries, and the definitive archive list."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .timecode import Timecode


@dataclass
class EDLEntry:
    """Represents a single entry (row) from an Edit Decision List (EDL).

    This contains the clip information extracted from AVID or Premiere Pro.
    """

    id: int
    name: str
    timecode_in: Timecode
    timecode_out: Timecode
    duration: Timecode
    source_start: Timecode
    source_end: Timecode
    reel: str = ""
    file_name: str = ""
    track: str = ""
    audio_channels: str = ""
    comment: str = ""

    @classmethod
    def from_dict(cls, data: dict, fps: int = 25) -> EDLEntry:
        """Create an EDLEntry from a dictionary (typically from a pandas row).

        Args:
            data: Dictionary containing EDL row data
            fps: Frame rate for timecode parsing

        Returns:
            EDLEntry instance
        """
        return cls(
            id=int(data.get("id", 0)),
            name=str(data.get("name", "")),
            timecode_in=Timecode.from_string(str(data.get("timecode_in", "00:00:00:00")), fps),
            timecode_out=Timecode.from_string(str(data.get("timecode_out", "00:00:00:00")), fps),
            duration=Timecode.from_string(str(data.get("duration", "00:00:00:00")), fps),
            source_start=Timecode.from_string(str(data.get("source_start", "00:00:00:00")), fps),
            source_end=Timecode.from_string(str(data.get("source_end", "00:00:00:00")), fps),
            reel=str(data.get("reel", "")),
            file_name=str(data.get("file_name", "")),
            track=str(data.get("track", "")),
            audio_channels=str(data.get("audio_channels", "")),
            comment=str(data.get("comment", "")),
        )


@dataclass
class SourceEntry:
    """Represents a source entry from the image researcher's archive list.

    This contains metadata about media sources including links and rights info.
    """

    name: str
    description: str = ""
    link: str = ""
    source: str = ""
    cost: str = ""
    rights_contact: str = ""
    todo_notes: str = ""
    source_in_frame: str = ""
    credits: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> SourceEntry:
        """Create a SourceEntry from a dictionary (typically from a pandas row).

        Args:
            data: Dictionary containing source row data

        Returns:
            SourceEntry instance
        """
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            link=str(data.get("link", "")),
            source=str(data.get("source", "")),
            cost=str(data.get("cost", "")),
            rights_contact=str(data.get("rights_contact", "")),
            todo_notes=str(data.get("todo_notes", "")),
            source_in_frame=str(data.get("source_in_frame", "")),
            credits=str(data.get("credits", "")),
        )


@dataclass
class DefEntry:
    """Represents an entry in the definitive archive list.

    This combines EDL timing information with source metadata.
    """

    name: str
    timecode_in: Timecode
    duration: Timecode
    description: str = ""
    link: str = ""
    source: str = ""
    cost: str = ""
    rights_contact: str = ""
    todo_notes: str = ""
    source_in_frame: str = ""
    credits: str = ""
    # Additional fields from combining
    source_start: Optional[Timecode] = None
    source_end: Optional[Timecode] = None

    @classmethod
    def from_edl_and_source(
        cls,
        edl: EDLEntry,
        source: Optional[SourceEntry] = None
    ) -> DefEntry:
        """Create a DefEntry by combining EDL timing with source metadata.

        Args:
            edl: EDL entry with timing information
            source: Optional source entry with metadata (if not found, fields are empty)

        Returns:
            DefEntry instance
        """
        entry = cls(
            name=edl.name,
            timecode_in=edl.timecode_in,
            duration=edl.duration,
            source_start=edl.source_start,
            source_end=edl.source_end,
        )

        if source:
            entry.description = source.description
            entry.link = source.link
            entry.source = source.source
            entry.cost = source.cost
            entry.rights_contact = source.rights_contact
            entry.todo_notes = source.todo_notes
            entry.source_in_frame = source.source_in_frame
            entry.credits = source.credits

        return entry

    def to_dict(self, include_frames: bool = False) -> dict:
        """Convert DefEntry to dictionary for output.

        Args:
            include_frames: If True, include frame-level precision in timecodes (HH:MM:SS:FF).
                          If False (default), round to seconds (HH:MM:SS).

        Returns:
            Dictionary representation of the entry
        """
        # Choose formatting method based on include_frames flag
        tc_format = lambda tc: tc.to_string() if include_frames else tc.to_string_rounded()

        return {
            "TC in": tc_format(self.timecode_in),
            "Duur": tc_format(self.duration),
            "Bestandsnaam": self.name,
            "Omschrijving": self.description,
            "Link": self.link,
            "Bron": self.source,
            "rechten / contact": self.rights_contact,
            "to do": self.todo_notes,
            "Prijs NL": self.cost,
            "Prijs sales": "",  # Not in source, kept for format compatibility
            "Bron in beeld": self.source_in_frame,
            "Aftiteling": self.credits,
            "Bron TC in": tc_format(self.source_start) if self.source_start else "",
            "Bron TC uit": tc_format(self.source_end) if self.source_end else "",
        }
