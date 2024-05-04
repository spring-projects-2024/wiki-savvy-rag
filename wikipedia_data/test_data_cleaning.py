from utils import (
    remove_template_tags,
    remove_table_tags,
    remove_wiki_tags,
    remove_square_brackets_around_links,
    extract_xml_tags,
)
import pytest


@pytest.mark.parametrize(
    "input, expected",
    [
        ("{{this should be removed}}this should be kept", "this should be kept"),
        ("{{this should be removed}}", ""),
        (
            "<math>{{this should be kept because it's within latex}}</math>{{but this should not}}",
            "<math>{{this should be kept because it's within latex}}</math>",
        ),
    ],
)
def test_remove_template_tags(input, expected):
    assert remove_template_tags(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            "<math>{|this should be kept because it's within latex|}</math>{|but this should not|}",
            "<math>{|this should be kept because it's within latex|}</math>",
        ),
        (
            "{|This should be removed {|as well as this|} |}leaving only this",
            "leaving only this",
        ),
        (
            "{|This should be removed <math> {|as well as this, despite being in latex|} </math> |}leaving only this{|and not this|}",
            "leaving only this",
        ),
        (
            "stuff{|this input is not well formed {|because a closing tag is missing |} but we do not want to discard everything in that case",
            "stuff{|this input is not well formed {|because a closing tag is missing |} but we do not want to discard everything in that case",
        ),
    ],
)
def test_remove_table_tags(input, expected):
    assert remove_table_tags(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("[[word: this should be removed]]this should be kept", "this should be kept"),
        (
            "[[ class: this should be removed]]this should be kept",
            "this should be kept",
        ),
        (
            "[[class : this should be removed]]this should be kept",
            "this should be kept",
        ),
        (
            "[[ class  : this should be removed]]this should be kept",
            "this should be kept",
        ),
        ("[[this should be kept]]", "[[this should be kept]]"),
        ("[[ this should : be kept]]", "[[ this should : be kept]]"),
        ("[[ this: [[should be removed]] removed]]kept", "kept"),
        ("[[ this [[should[[be]]]] kept]]", "[[ this [[should[[be]]]] kept]]"),
        (
            "[[ this: is not [[ well formed: stuff here]]",
            "[[ this: is not [[ well formed: stuff here]]",
        ),
    ],
)
def test_remove_wiki_tags(input, expected):
    assert remove_wiki_tags(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("[[Title]]", "Title"),
        ("[[Title|Alias]]", "Title"),
        ("<math>[[Title|Alias]]</math>", "<math>[[Title|Alias]]</math>"),
        ("[[First|[[Second]]", "First"),
        # ("[[[[Title]]]]", ""),
    ],
)
def test_remove_square_brackets_around_links(input, expected):
    assert remove_square_brackets_around_links(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            "<page>\n<text metadati>text</text>\n<title>title</title>\n</page>",
            "<page>\n<title>\ntitle\n</title>\n<text>\ntext\n</text>\n</page>",
        ),
        (
            "<text metadata>text</text>",
            "<page>\n<title>\n\n</title>\n<text>\ntext\n</text>\n</page>",
        ),
        (
            "<text> We talk about html tags such as </text>, <text>, keep this</text>",
            "<page>\n<title>\n\n</title>\n<text>\n We talk about html tags such as </text>, <text>, keep this\n</text>\n</page>",
        ),
    ],
)
def test_extract_xml_tags(input, expected):
    assert extract_xml_tags(input) == expected
