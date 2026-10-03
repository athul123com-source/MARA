"""Fix FIRST REVIEW new.pptx -> FIRST REVIEW FINAL.pptx (original untouched).

Fixes:
1. Footers: uniform numbering scheme (content slides = position-1, matching
   slides 2-13 majority), uniform gaps, no trailing spaces, 'No:9' -> 'No: 9',
   and a footer added to the THANK YOU slide (18).
2. Literature review tables (slides 6-9): replace IoT/irrigation papers with
   real RAG / multi-agent / hallucination papers relevant to MARA.
3. Work plan (slide 15): remove advanced-feature claims (embedding
   fine-tuning, cross-paper synthesis); fix 'KS KISHNAPRIYA' -> 'K S KRISHNAPRIYA'.
4. Resources (slide 14): 'React + Vite' -> 'Streamlit' (matches plan).
5. Feasibility (slide 16): complete the truncated sentence.
6. Typos: 'research plan..' -> 'research plan.' (slide 10), remove stray '..'.
7. Remove empty trailing table rows on slides 6-7.
"""
import copy
import re

from pptx import Presentation
from pptx.oxml.ns import qn

SRC = r"C:\Users\athul\Downloads\FIRST REVIEW new.pptx"
DST = r"C:\Users\athul\Downloads\FIRST REVIEW FINAL.pptx"

prs = Presentation(SRC)
slides = list(prs.slides)


# ---------------------------------------------------------------- helpers
def single_run_text(tf, text):
    """Put `text` into the first run of the first paragraph; delete other runs."""
    para = tf.paragraphs[0]
    runs = para.runs
    assert runs, "paragraph has no runs"
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def set_footer(shape, number):
    tf = shape.text_frame
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    single_run_text(tf, footer_text(number))


def footer_text(number):
    return f"Date{GAP1}IESCE{GAP2}Slide. No: {number}"


def delete_run(run):
    run._r.getparent().remove(run._r)


def cell_set(cell, text):
    """Replace a table cell's text, keeping first-paragraph/run formatting."""
    tf = cell.text_frame
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    single_run_text(tf, text)


def footer_box(slide):
    """The combined 'Date ... IESCE ... Slide. No: n' text box, if present."""
    for shape in slide.shapes:
        if shape.has_text_frame:
            t = shape.text_frame.text
            if "Date" in t and "IESCE" in t and "Slide. No" in t:
                return shape
    return None


# ------------------------------------------------- 1. capture footer gaps
ref = footer_box(slides[1])  # slide 2 is the reference style
m = re.match(r"^Date( +)IESCE( +)Slide\. No: ?\d+\s*$", ref.text_frame.text)
assert m, "unexpected footer format"
GAP1, GAP2 = m.group(1), m.group(2)

# text-box footer slides -> numbering scheme (position-1, majority style)
FOOTER_NUMBERS = {2: 1, 3: 2, 4: 3, 5: 4, 10: 9, 11: 10, 12: 11, 13: 12,
                  14: 13, 15: 14, 16: 15, 17: 16}
for pos, num in FOOTER_NUMBERS.items():
    box = footer_box(slides[pos - 1])
    assert box is not None, f"slide {pos}: footer box missing"
    set_footer(box, num)

# slides 6-9 already use correct footer placeholders/numbering (5..8) - keep.
# Slide 18 (THANK YOU): clone slide 17's footer box and number it 17.
src_footer = footer_box(slides[16])
new_sp = copy.deepcopy(src_footer._element)
cnvpr = new_sp.find(qn("p:nvSpPr") + "/" + qn("p:cNvPr"))
cnvpr.set("id", "9018")
cnvpr.set("name", "Footer slide 18")
slides[17].shapes._spTree.append(new_sp)
set_footer(footer_box(slides[17]), 17)
print("footers done")

# --------------------------------------- 2. literature review replacement
PAPERS = {
    # slide position -> list of (slno, name, outcome, author/details) per body row
    6: [
        ("1.",
         "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
         "Introduces Retrieval-Augmented Generation (RAG), which grounds an "
         "LLM's answers in passages retrieved from a knowledge base, sharply "
         "reducing hallucination and enabling verifiable, cited outputs.",
         "Author: Patrick Lewis et al. Published on: 2020, Advances in "
         "Neural Information Processing Systems (NeurIPS), Volume 33, "
         "Pages 9459-9474."),
        ("2.",
         "Retrieval-Augmented Generation for Large Language Models: A Survey",
         "Surveys naive, advanced and modular RAG designs and shows that "
         "retrieval quality directly controls answer factuality, guiding "
         "the retrieval design used in MARA.",
         "Author: Yunfan Gao et al. Published on: 2023, arXiv preprint "
         "arXiv:2312.10997."),
    ],
    7: [
        ("3.",
         "Self-RAG: Learning to Retrieve, Generate, and Critique through "
         "Self-Reflection",
         "Trains a model to decide when to retrieve and to critique its own "
         "drafts with reflection tokens, improving factual precision and "
         "citation accuracy - the closest published relative of MARA's "
         "Critic Agent.",
         "Authors: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, "
         "Hannaneh Hajishirzi. Published on: 2024, International Conference "
         "on Learning Representations (ICLR)."),
        ("4.",
         "Retrieval Augmentation Reduces Hallucination in Conversation",
         "Shows that grounding chatbot responses in retrieved evidence "
         "measurably reduces hallucinated content, supporting retrieval "
         "plus verification over pure generation.",
         "Authors: Kurt Shuster, Spencer Poff, Douwe Kiela, Jason Weston. "
         "Published on: 2021, Findings of EMNLP, Pages 3454-3465."),
    ],
    8: [
        ("5.",
         "DSPy: Compiling Declarative Language Model Calls into "
         "Self-Improving Pipelines",
         "Presents a framework that compiles declarative multi-step "
         "retrieve-verify-generate pipelines and shows structured agent "
         "pipelines outperform one-shot prompting on knowledge tasks.",
         "Author: Omar Khattab et al. Published on: 2024, International "
         "Conference on Learning Representations (ICLR), "
         "arXiv:2310.12823."),
        ("6.",
         "From Local to Global: A Graph RAG Approach to Query-Focused "
         "Summarization",
         "Proposes GraphRAG, which combines agent-style retrieval with "
         "community summaries for query-focused summarization over large "
         "document corpora, improving coverage and answer groundedness.",
         "Authors: Darren Edge et al. Published on: 2024, arXiv preprint "
         "arXiv:2404.16130."),
    ],
    9: [
        ("7.",
         "A Survey on Hallucination in Large Language Models: Principles, "
         "Taxonomy, Challenges, and Open Questions",
         "Provides a taxonomy of LLM hallucinations and mitigation "
         "techniques, including retrieval grounding and verification, "
         "directly motivating MARA's claim-level fact-checking stage.",
         "Authors: Lei Huang et al. Published on: 2024, ACM Transactions "
         "on Information Systems, arXiv:2311.05232."),
    ],
}

for pos, papers in PAPERS.items():
    slide = slides[pos - 1]
    table = next(sh.table for sh in slide.shapes
                 if getattr(sh, "has_table", False) and sh.has_table)
    body_rows = list(table.rows)[1:]
    assert len(body_rows) >= len(papers), f"slide {pos}: not enough rows"
    for row, paper in zip(body_rows, papers):
        for cell, text in zip(row.cells, paper):
            cell_set(cell, text)
    # drop fully-empty trailing rows (template leftovers)
    for row in list(table.rows)[1 + len(papers):]:
        if all(not c.text.strip() for c in row.cells):
            row._tr.getparent().remove(row._tr)
print("literature tables done")

# --------------------------------------------------- 3. work plan fixes
wp_table = next(sh.table for sh in slides[14].shapes
                if getattr(sh, "has_table", False) and sh.has_table)
# row 0 = Athul: strip fine-tuning + cross-paper synthesis claims
tf = wp_table.cell(0, 1).text_frame
runs = tf.paragraphs[0].runs  # bullets separated by <a:br/> we keep
runs[0].text = ("\u2022 Lead the development of the core multi-agent pipeline "
                "(Planner, Retriever, Critic, Writer) and orchestrate it "
                "using LangGraph.")
delete_run(runs[2]); delete_run(runs[1])
runs = tf.paragraphs[0].runs  # re-query after deletions
runs[1].text = ("\u2022 Build the FastAPI backend that serves the MARA "
                "pipeline and connects the web front end.")
for r in runs[4:8]:
    delete_run(r)
runs = tf.paragraphs[0].runs
runs[2].text = ("\u2022 Integrate the Critic-based verification and "
                "citation-backed answer generation.")
delete_run(runs[3])
# row 4 = Krishnapriya: remove embedding fine-tuning task
tf = wp_table.cell(4, 1).text_frame
runs = tf.paragraphs[0].runs
runs[1].text = ("\u2022 Assist in preparing and organizing research datasets "
                "used for retrieval and evaluation.")
delete_run(runs[3]); delete_run(runs[2])
# name spelling consistency with title slide
cell_set(wp_table.cell(4, 0), "K S KRISHNAPRIYA")
print("work plan done")

# ------------------------------------------ 4. resources: Streamlit swap
for sh in slides[13].shapes:
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                if run.text.strip() == "React + Vite":
                    run.text = "Streamlit "
print("resources done")

# ------------------------------ 5. feasibility: complete truncated line
for sh in slides[15].shapes:
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            runs = para.runs
            if runs and runs[-1].text.endswith("multi-agent "):
                runs[-1].text = "LLMs, RAG, and multi-agent frameworks"
                tail = copy.deepcopy(runs[0]._r)
                tail.find(qn("a:t")).text = (" already studied in coursework, "
                                             "keeping the technical risk low.")
                runs[-1]._r.addnext(tail)
print("feasibility done")

# ------------------------------------------------------- 6. typo fixes
for sh in slides[9].shapes:  # slide 10 'research plan..'
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            runs = para.runs
            for r_i, run in enumerate(runs):
                if run.text == "." and r_i > 0 and runs[r_i - 1].text.endswith("."):
                    delete_run(run)
                    break
print("typos done")

prs.save(DST)
print("saved:", DST)
