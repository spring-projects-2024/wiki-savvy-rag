from utils import remove_template_tags, remove_table_tags, remove_wiki_tags, remove_square_brackets_around_links
import pytest


@pytest.mark.parametrize("input, expected", [
    ("{{this should be removed}}this should be kept", "this should be kept"),
    ("{{this should be removed}}", ""),
    ("<math>{{this should be kept because it's within latex}}</math>{{but this should not}}",
     "<math>{{this should be kept because it's within latex}}</math>"),
])
def test_remove_template_tags(input, expected):
    assert remove_template_tags(input) == expected


@pytest.mark.parametrize("input, expected", [
    ("<math>{|this should be kept because it's within latex|}</math>{|but this should not|}",
     "<math>{|this should be kept because it's within latex|}</math>"),
    ("{|This should be removed {|as well as this|} |}leaving only this", "leaving only this"),
    (
    "{|This should be removed <math> {|as well as this, despite being in latex|} </math> |}leaving only this{|and not this|}",
    "leaving only this"),
])
def test_remove_table_tags(input, expected):
    assert remove_table_tags(input) == expected


@pytest.mark.parametrize("input, expected", [
    ("[[word: this should be removed]]this should be kept", "this should be kept"),
    ("[[ class: this should be removed]]this should be kept", "this should be kept"),
    ("[[class : this should be removed]]this should be kept", "this should be kept"),
    ("[[ class  : this should be removed]]this should be kept", "this should be kept"),
    ("[[this should be kept]]", "[[this should be kept]]"),
    ("[[ this should : be kept]]", "[[ this should : be kept]]"),
    ("[[ this: [[should be removed]] removed]]kept", "kept"),
    ("[[ this [[should[[be]]]] kept]]", "[[ this [[should[[be]]]] kept]]"),
])
def test_remove_wiki_tags(input, expected):
    assert remove_wiki_tags(input) == expected


@pytest.mark.parametrize("input, expected", [
    ("[[Title]]", "Title"),
    ("[[Title|Alias]]", "Title"),
    ("<math>[[Title|Alias]]</math>", "<math>[[Title|Alias]]</math>"),
])
def test_remove_square_brackets_around_links(input, expected):
    assert remove_square_brackets_around_links(input) == expected
