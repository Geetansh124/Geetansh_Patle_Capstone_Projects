"""Compile Capstone Markdown Reports and Presentations to PDF using PyMuPDF."""
from pathlib import Path
import pymupdf


def markdown_to_pdf(md_path: Path, pdf_path: Path, is_presentation: bool = False):
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return

    text = md_path.read_text(encoding="utf-8")
    doc = pymupdf.open()
    
    # Dimensions: A4 Portrait (595x842) for report, Landscape (842x595) for presentation
    page_w, page_h = (842, 595) if is_presentation else (595, 842)
    margin = 50

    if is_presentation:
        raw_slides = text.split("### Slide ")
        for idx, slide_content in enumerate(raw_slides):
            if not slide_content.strip():
                continue
            slide_text = ("Slide " + slide_content) if idx > 0 else slide_content
            page = doc.new_page(width=page_w, height=page_h)
            
            # Header accent bar
            page.draw_rect(pymupdf.Rect(0, 0, page_w, 8), color=(0.15, 0.45, 0.85), fill=(0.15, 0.45, 0.85))
            
            lines = slide_text.strip().split("\n")
            title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip()
            
            # Draw Title
            page.insert_text((margin, margin + 25), title, fontsize=20, color=(0.1, 0.2, 0.4))
            
            # Draw Body
            page.insert_textbox(
                pymupdf.Rect(margin, margin + 45, page_w - margin, page_h - margin - 25),
                body,
                fontsize=11,
                color=(0.15, 0.15, 0.15),
                lineheight=1.4
            )
            
            # Footer
            footer_text = f"AIML Capstone Presentation | Page {len(doc)}"
            page.insert_text((margin, page_h - 25), footer_text, fontsize=9, color=(0.5, 0.5, 0.5))
    else:
        paragraphs = text.split("\n\n")
        current_page = doc.new_page(width=page_w, height=page_h)
        y = margin

        # Draw top accent bar
        current_page.draw_rect(pymupdf.Rect(0, 0, page_w, 6), color=(0.15, 0.45, 0.85), fill=(0.15, 0.45, 0.85))

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Determine styling
            if para.startswith("# "):
                fontsize = 17
                color = (0.08, 0.18, 0.38)
                content = para.replace("# ", "").strip()
                space_after = 16
            elif para.startswith("## "):
                fontsize = 13
                color = (0.12, 0.25, 0.5)
                content = para.replace("## ", "").strip()
                space_after = 12
            elif para.startswith("### "):
                fontsize = 11.5
                color = (0.2, 0.3, 0.45)
                content = para.replace("### ", "").strip()
                space_after = 10
            elif para.startswith("```"):
                fontsize = 8.5
                color = (0.2, 0.2, 0.2)
                content = para.replace("```text", "").replace("```", "").strip()
                space_after = 12
            else:
                fontsize = 9.5
                color = (0.15, 0.15, 0.15)
                content = para
                space_after = 10

            line_count = max(1, len(content) // 75 + content.count("\n") + 1)
            estimated_h = line_count * (fontsize * 1.35) + space_after

            if y + estimated_h > (page_h - margin - 30):
                current_page.insert_text(
                    (margin, page_h - 25),
                    f"Geetansh Patle — Capstone Technical Report | Page {len(doc)}",
                    fontsize=8.5,
                    color=(0.5, 0.5, 0.5)
                )
                current_page = doc.new_page(width=page_w, height=page_h)
                current_page.draw_rect(pymupdf.Rect(0, 0, page_w, 6), color=(0.15, 0.45, 0.85), fill=(0.15, 0.45, 0.85))
                y = margin

            target_rect = pymupdf.Rect(margin, y, page_w - margin, min(page_h - margin, y + estimated_h + 10))
            current_page.insert_textbox(
                target_rect,
                content,
                fontsize=fontsize,
                color=color,
                lineheight=1.35
            )
            y += estimated_h

        current_page.insert_text(
            (margin, page_h - 25),
            f"Geetansh Patle — Capstone Technical Report | Page {len(doc)}",
            fontsize=8.5,
            color=(0.5, 0.5, 0.5)
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    page_count = len(doc)
    doc.close()
    print(f"Generated {pdf_path} ({page_count} pages)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    
    # AIML-02
    markdown_to_pdf(base / "AIML-02" / "REPORT.md", base / "AIML-02" / "report.pdf", is_presentation=False)
    markdown_to_pdf(base / "AIML-02" / "PRESENTATION.md", base / "AIML-02" / "presentation.pdf", is_presentation=True)
    
    # AIML-03
    markdown_to_pdf(base / "AIML-03" / "REPORT.md", base / "AIML-03" / "report.pdf", is_presentation=False)
    markdown_to_pdf(base / "AIML-03" / "PRESENTATION.md", base / "AIML-03" / "presentation.pdf", is_presentation=True)
