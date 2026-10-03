"""Inspect run structure of shapes targeted for editing."""
from pptx import Presentation

PPTX = r"C:\Users\athul\Downloads\FIRST REVIEW new.pptx"
prs = Presentation(PPTX)


def dump_runs(label, tf):
    print(f"\n--- {label} ---")
    for p_i, para in enumerate(tf.paragraphs):
        for r_i, run in enumerate(para.runs):
            rpr = run._r.find('{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')
            print(f"  p{p_i}r{r_i}: {run.text!r}  (has_rPr={rpr is not None})")
        brs = para._p.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}br')
        if brs:
            print(f"  p{p_i}: <a:br/> x{len(brs)}")


# Slide 10 'plan..' typo
s10 = prs.slides[9]
for sh in s10.shapes:
    if sh.has_text_frame and 'research plan' in sh.text_frame.text:
        dump_runs("SLIDE 10 proposed system", sh.text_frame)

# Slide 14 resources
s14 = prs.slides[13]
for sh in s14.shapes:
    if sh.has_text_frame and 'React' in sh.text_frame.text:
        dump_runs("SLIDE 14 resources", sh.text_frame)

# Slide 16 feasibility
s16 = prs.slides[15]
for sh in s16.shapes:
    if sh.has_text_frame and 'multi-agent' in sh.text_frame.text:
        dump_runs("SLIDE 16 feasibility", sh.text_frame)

# Slide 15 work plan table cell run structure
s15 = prs.slides[14]
for sh in s15.shapes:
    if getattr(sh, "has_table", False) and sh.has_table:
        for r_i, row in enumerate(sh.table.rows):
            name = row.cells[0].text
            if 'ATHUL' in name or 'KISHNAPRIYA' in name or 'KRISHNAPRIYA' in name:
                dump_runs(f"SLIDE 15 row{r_i} NAME CELL", row.cells[0].text_frame)
                dump_runs(f"SLIDE 15 row{r_i} TASK CELL", row.cells[1].text_frame)

# Slide 6 literature table: header + one body cell
s6 = prs.slides[5]
for sh in s6.shapes:
    if getattr(sh, "has_table", False) and sh.has_table:
        tbl = sh.table
        dump_runs("SLIDE 6 header row0c1", tbl.cell(0, 1).text_frame)
        dump_runs("SLIDE 6 row1c1 (paper name)", tbl.cell(1, 1).text_frame)
        dump_runs("SLIDE 6 row1c2 (outcome)", tbl.cell(1, 2).text_frame)
        dump_runs("SLIDE 6 row1c3 (author)", tbl.cell(1, 3).text_frame)

# Character codes of suspected bullet/dash chars
print("\n=== CHAR CODES ===")
s4 = prs.slides[3]
for sh in s4.shapes:
    if sh.has_text_frame and 'Save research time' in sh.text_frame.text:
        t = sh.text_frame.text
        print("slide4 sample:", [(c, hex(ord(c))) for c in t[:40] if ord(c) > 126])
s14t = [sh.text_frame.text for sh in s14.shapes if sh.has_text_frame and 'GB RAM' in sh.text_frame.text][0]
print("slide14 RAM line:", [(c, hex(ord(c))) for c in s14t.split('\n') if 'RAM' in c for c in c if ord(c) > 126])
s15t = [sh for sh in s15.shapes if getattr(sh, 'has_table', False) and sh.has_table][0]
c = s15t.table.cell(0, 1).text
print("slide15 bullets:", [(ch, hex(ord(ch))) for ch in c if ord(ch) > 126])
