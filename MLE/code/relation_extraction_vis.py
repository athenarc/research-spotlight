from sparknlp.base import Annotation
from sparknlp_display import RelationExtractionVisualizer
import ipywidgets as widgets
from IPython.display import display

index = 0
examples = []

output = widgets.Output()
btn_next = widgets.Button(description="→")
btn_prev = widgets.Button(description="←")
btn_first = widgets.Button(description="↺")

vis = RelationExtractionVisualizer()


def render(data=None):
    global index, examples

    if data is not None:
        examples = data
        index = 0

    output.clear_output()

    if not examples:
        return

    text = examples[index]

    result = build_mock(text)

    with output:
        vis.display(
            result,
            relation_col="relations",
            document_col="document",
            show_relations=True
        )
        print(f"{index + 1} / {len(examples)}")


def build_mock(text):
    words = text[0].split()

    document = [Annotation("DOCUMENT", 0, len(text[0]) - 1, text[0], {}, None)]

    tokens = []
    cursor = 0
    for word in words:
        start = text[0].find(word, cursor)
        end = start + len(word) - 1
        tokens.append(Annotation("token", start, end, word, {}, None))
        cursor = end + 1

    relations = [
        Annotation(
            "RELATION",
            0,
            len(text[0]) - 1,
            text[1],
            {
                "entity1": text[2][2],
                "entity1_begin": text[2][0],
                "entity1_end": text[2][1],
                "chunk1": text[0][text[2][0]:text[2][1]],

                "entity2": text[3][2],
                "entity2_begin": text[3][0],
                "entity2_end": text[3][1],
                "chunk2": text[0][text[3][0]:text[3][1]],
            },
            None
        )
    ]

    return {
        "document": document,
        "tokens": [],
        "relations": relations
    }


def first_click(b):
    global index
    index = 0
    render()


def next_click(b):
    global index
    index = (index + 1) % len(examples)
    render()


def prev_click(b):
    global index
    index = (index - 1) % len(examples)
    render()


btn_next.on_click(next_click)
btn_prev.on_click(prev_click)
btn_first.on_click(first_click)
