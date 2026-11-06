from __future__ import annotations

from app.prompts import is_document_overview


def test_document_overview_phrasings_are_recognized():
    assert is_document_overview("What do you understand by the given document?")
    assert is_document_overview("Give me an overview of this report")
    assert is_document_overview("Summarize the uploaded PDF")
    assert not is_document_overview("How many days of paid leave are provided?")
