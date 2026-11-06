import pytest

from app.chunking import chunk_pages, split_text

TEXT = "\n".join(f"Line {i}: " + "word " * 12 for i in range(40))


def test_chunks_respect_size_and_cover_all_text():
    chunks = split_text(TEXT, 300, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    joined = "\n".join(chunks)
    for i in range(40):
        assert f"Line {i}:" in joined


def test_overlap_repeats_trailing_lines():
    chunks = split_text(TEXT, 300, 100)
    first_tail = chunks[0].splitlines()[-1]
    assert first_tail in chunks[1]


def test_zero_overlap_has_no_repeated_lines():
    chunks = split_text(TEXT, 300, 0)
    seen = [line for c in chunks for line in c.splitlines()]
    assert len(seen) == len(set(seen))


def test_long_line_is_split_on_sentences_then_hard_split():
    sentence_text = " ".join(f"Sentence number {i} is here." for i in range(50))
    assert all(len(c) <= 200 for c in split_text(sentence_text, 200, 20))
    assert all(len(c) <= 100 for c in split_text("x" * 1000, 100, 10))


def test_pages_keep_page_numbers_and_global_index():
    chunks = chunk_pages([(1, TEXT), (2, "short page")], 300, 50)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2 and chunks[0].page == 1


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)
