import os
import re
import base64
import argparse
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


SLIDE_NOTE_INSTRUCTIONS = """
You are a technical training slide reader.

Your task:
Read each slide image and the extracted text.
Create rich slide notes that will later be used to write a narration script.

Rules:
- Do not write the final narration yet.
- Preserve important technical terms.
- If a slide has diagrams, explain what the diagram means.
- If a slide has little text but many visuals, infer the teaching point from the visuals.
- Mention every slide exactly once.
- Write in English unless the slide is in another language.

Output format:

[Slide X]
Title: ...
Visible content: ...
Visual explanation: ...
Teaching point: ...
"""


FINAL_SCRIPT_INSTRUCTIONS = """
You are a professional technical narration script writer.

You must create a narration script from slide notes by following the sample script style.

Hard requirements:
1. Follow the sample TXT format closely.
2. Include title lines at the top.
3. Include:
   === NARRATION SCRIPT ===
   [INTRO]
   [WHY]
   [WHAT]
   [HOW]
   [FLOW]
   [PRACTICE]
   [SUMMARY]
   === FULL NARRATION ===
4. Mention every slide using [Slide X].
5. Do not copy bullet points mechanically.
6. Convert slide content into natural voice-over narration.
7. Keep important technical terms in English.
8. Make the script sound like a real classroom training narration.
9. Do not invent unrelated content.
10. The FULL NARRATION section must be a smooth, continuous speech version.
"""


def read_text_file(path: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def render_page_to_data_url(page, zoom: float = 1.6) -> str:
    """
    Render một trang PDF thành ảnh PNG base64 để model nhìn được cả text và hình.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_text_from_page(page) -> str:
    text = page.get_text("text").strip()
    return text if text else "(No extracted text)"


def get_output_text(response) -> str:
    """
    Lấy text output từ response theo cách an toàn.
    """
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") == "output_text":
                chunks.append(content.text)
    return "\n".join(chunks).strip()


def make_slide_notes(
    client: OpenAI,
    pdf_path: str,
    model: str,
    chunk_size: int = 6,
    zoom: float = 1.6,
    max_tokens_per_chunk: int = 5000,
) -> str:
    """
    Bước 1:
    Chia PDF thành từng cụm slide.
    Mỗi cụm gửi text + ảnh slide cho model để tạo slide notes.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    all_notes = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)

        content = [
            {
                "type": "input_text",
                "text": (
                    SLIDE_NOTE_INSTRUCTIONS
                    + f"\n\nYou are reading slides {start + 1} to {end} of {total_pages}.\n"
                    + "Below are extracted texts and rendered slide images.\n"
                ),
            }
        ]

        for i in range(start, end):
            page = doc[i]
            slide_number = i + 1
            extracted_text = extract_text_from_page(page)
            image_url = render_page_to_data_url(page, zoom=zoom)

            content.append(
                {
                    "type": "input_text",
                    "text": f"\n\n--- Extracted text for [Slide {slide_number}] ---\n{extracted_text}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                }
            )

        print(f"Đang đọc slide {start + 1} đến {end}...")

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0.2,
            max_output_tokens=max_tokens_per_chunk,
        )

        notes = get_output_text(response)
        all_notes.append(notes)

    doc.close()
    return "\n\n".join(all_notes)


def create_final_script(
    client: OpenAI,
    slide_notes: str,
    sample_script: str,
    model: str,
    total_slides: int,
    max_output_tokens: int = 18000,
) -> str:
    """
    Bước 2:
    Dùng slide notes + script mẫu để viết final narration script.
    """
    prompt = f"""
You must write a new narration script for a slide deck.

Total slides: {total_slides}

Use this sample script only to learn FORMAT, STRUCTURE, SECTION STYLE, and WRITING TONE:

<<< SAMPLE SCRIPT START >>>
{sample_script}
<<< SAMPLE SCRIPT END >>>

Now use these slide notes as the actual content source:

<<< SLIDE NOTES START >>>
{slide_notes}
<<< SLIDE NOTES END >>>

Write the final result now.
"""

    print("Đang tạo final narration script...")

    response = client.responses.create(
        model=model,
        instructions=FINAL_SCRIPT_INSTRUCTIONS,
        input=prompt,
        temperature=0.25,
        max_output_tokens=max_output_tokens,
    )

    return get_output_text(response)


def find_missing_slides(script: str, total_slides: int) -> list[int]:
    found = set(int(x) for x in re.findall(r"\[Slide\s+(\d+)\]", script))
    return [i for i in range(1, total_slides + 1) if i not in found]


def repair_script_if_needed(
    client: OpenAI,
    script: str,
    slide_notes: str,
    model: str,
    total_slides: int,
    max_output_tokens: int = 18000,
) -> str:
    """
    Nếu script bị thiếu [Slide X], yêu cầu model sửa lại.
    """
    missing = find_missing_slides(script, total_slides)

    if not missing:
        print("Kiểm tra OK: không thiếu slide.")
        return script

    print(f"Cảnh báo: script đang thiếu các slide: {missing}")
    print("Đang yêu cầu model sửa lại script...")

    prompt = f"""
The narration script below is missing these slide references:
{missing}

Repair the script so that every slide from 1 to {total_slides} appears at least once.
Do not shorten the script too much.
Keep the same required format.

<<< CURRENT SCRIPT START >>>
{script}
<<< CURRENT SCRIPT END >>>

Use these slide notes to restore missing slide content:

<<< SLIDE NOTES START >>>
{slide_notes}
<<< SLIDE NOTES END >>>
"""

    response = client.responses.create(
        model=model,
        instructions=FINAL_SCRIPT_INSTRUCTIONS,
        input=prompt,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )

    repaired = get_output_text(response)
    still_missing = find_missing_slides(repaired, total_slides)

    if still_missing:
        print(f"Vẫn còn thiếu slide: {still_missing}")
    else:
        print("Đã sửa xong: không thiếu slide.")

    return repaired


def main():
    print("SCRIPT STARTED")
    parser = argparse.ArgumentParser(
        description="Generate narration script from slide PDF using a sample TXT script."
    )

    parser.add_argument("--pdf", required=True, help="Đường dẫn tới file slide PDF")
    parser.add_argument("--sample", required=True, help="Đường dẫn tới file script mẫu TXT")
    parser.add_argument("--out", default="generated_script.txt", help="File output TXT")
    parser.add_argument("--notes-out", default="slide_notes.txt", help="File lưu slide notes trung gian")
    parser.add_argument("--model", default="gpt-4.1", help="Model dùng để tạo script")
    parser.add_argument("--chunk-size", type=int, default=6, help="Số slide gửi mỗi lần")
    parser.add_argument("--zoom", type=float, default=1.6, help="Độ nét render slide")
    parser.add_argument("--no-repair", action="store_true", help="Không tự sửa nếu thiếu slide")

    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Thiếu OPENAI_API_KEY. Hãy tạo file .env hoặc set biến môi trường.")

    pdf_path = Path(args.pdf)
    sample_path = Path(args.sample)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")

    if not sample_path.exists():
        raise FileNotFoundError(f"Không tìm thấy TXT mẫu: {sample_path}")

    client = OpenAI()

    total_slides = get_pdf_page_count(str(pdf_path))
    sample_script = read_text_file(str(sample_path))

    print(f"PDF: {pdf_path}")
    print(f"Sample script: {sample_path}")
    print(f"Tổng số slide: {total_slides}")

    slide_notes = make_slide_notes(
        client=client,
        pdf_path=str(pdf_path),
        model=args.model,
        chunk_size=args.chunk_size,
        zoom=args.zoom,
    )

    Path(args.notes_out).write_text(slide_notes, encoding="utf-8")
    print(f"Đã lưu slide notes vào: {args.notes_out}")

    final_script = create_final_script(
        client=client,
        slide_notes=slide_notes,
        sample_script=sample_script,
        model=args.model,
        total_slides=total_slides,
    )

    if not args.no_repair:
        final_script = repair_script_if_needed(
            client=client,
            script=final_script,
            slide_notes=slide_notes,
            model=args.model,
            total_slides=total_slides,
        )

    Path(args.out).write_text(final_script, encoding="utf-8")
    print(f"Đã tạo script xong: {args.out}")


if __name__ == "__main__":
    main()