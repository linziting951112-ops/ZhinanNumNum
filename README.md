# Project Title

ZhinanNumNum – A RAG-Based Restaurant Recommender for NCCU

## Project Description

ZhinanNumNum is a Retrieval-Augmented Generation (RAG) chatbot that helps National Chengchi University (NCCU) students and visitors decide where to eat near the campus.

Students often waste time deciding among the many small eateries around campus. ZhinanNumNum lets a user ask in natural language, for instance, by cuisine, budget, group size, vibe, or time of day, and then returns 2–4 tailored recommendations with address, opening hours, price range, and a signature dish.

* Data

We collected information on 82 restaurants near NCCU, including name, address, cuisine type, price level, opening hours, dine-in availability, and up to five Google customer reviews per restaurant. The raw data was scraped from Google Maps (via Google API), cleaned, and stored as nccu_restaurants_cleaned.csv.

* Method

The system uses a RAG architecture: each restaurant is embedded into a vector and stored in a vector database; at query time the most relevant restaurants are retrieved and passed to a large language model, which writes a friendly recommendation grounded in the retrieved data. This keeps answers factual and easy to update.

## Getting Started

* Prerequisites

Python 3.10+
An OpenAI API key (set as the environment variable OPENAI_API_KEY)

Dependencies (see requirements.txt)

* Install

pip install -r requirements.txt

* Run locally

export OPENAI_API_KEY="your-key-here"
python app.py

The app launches a Gradio web interface in the browser.

* Demo

The chatbot is deployed on Hugging Face Spaces: https://huggingface.co/spaces/chrisss00/ZhinanNumNum 

## File Structure

* app.py — On first launch it builds a persistent ChromaDB collection by reading each restaurant's PDF (document content) and the CSV (structured metadata), embedding them, and storing them. It then serves the chatbot, handling query expansion, time-based filtering, and LLM response generation.
* restaurant_pdfs — Each restaurant is represented as its own PDF document, generated from the CSV. The filename is prefixed with the restaurant's row index (e.g., 03_...pdf) so it can be matched back to its metadata in the CSV.
* nccu_restaurants_cleaned.csv — Provides the structured fields (cuisine, price, opening hours, etc.) used for filtering.

## Analysis

* System architecture (RAG, two phases)

1. Phase 1 — Build the vector database (runs once when setting up). Each restaurant's information is converted into a vector ("embedding") and stored in ChromaDB, our vector database.

2. Phase 2 — Query and response (runs every time a user asks). The user's question is also embedded, the most similar restaurants are retrieved from ChromaDB, and a language model turns the retrieved data into a natural-language recommendation.

* AI components

The system relies on three AI components working together. First, the embedding model, OpenAI text-embedding-3-small, turns each restaurant description and the user's query into comparable vectors. Second, the vector database, ChromaDB, stores these embeddings and quickly retrieves the restaurants most relevant to a given query. Third, the language model, OpenAI gpt-4o-mini, reads the retrieved restaurants and writes the final natural-language recommendation.

* Data pipeline: from CSV rows to per-restaurant PDF documents

Following our instructor's suggestion, we revised the data pipeline so that each restaurant is represented as its own PDF document, rather than concatenating its CSV columns into a single text string.

Before: every restaurant row was flattened into one text string (name, cuisine, address, price, hours, reviews) and embedded directly from the CSV.

After: we generate one PDF per restaurant (using fpdf2), then load each PDF with LangChain's PyPDFLoader and embed the extracted text. The structured metadata used for filtering (e.g., opening hours, price) is still read from the CSV and attached to each entry.

The motivation is that treating each restaurant as a self-contained document gives a cleaner, more complete unit of context for retrieval, and reflects a more realistic document-based RAG workflow. One restaurant maps to exactly one document and one retrieval result, which keeps recommendations clean and the metadata aligned.

* Retrieval and filtering design

Two challenges shaped our retrieval design:

1. Cuisine queries can be confused by review text. To strengthen cuisine signals, we expand the user's query with related terms (e.g., "Korean" → "Korean BBQ, bibimbap, kimchi…") before embedding it.

2. Opening-hours logic is unreliable when left to the LLM. gpt-4o-mini consistently failed at cross-midnight time arithmetic (e.g., a place open "6:00 PM – 1:00 AM"). We therefore moved all time validation into Python: when a query mentions a time, we

(a) increase the number of retrieved candidates (K = 50 instead of 10) so late-night venues surface, and 

(b) filter candidates with a Python function that correctly parses cross-midnight ranges, multi-period hours, and weekly closures, using the opening-hours metadata. Only verified-open restaurants are passed to the LLM.

## Results

We evaluated the system with a 10-question manual test across five dimensions: cuisine match, budget filtering, time logic, semantic understanding, and edge cases. For each query we compared the chatbot's response against the expected behavior and scored it Pass / Partial / Fail.

Representative findings:

  * Cuisine match — Correctly returns only the relevant restaurants (e.g., a request for Korean food returns the three Korean restaurants in the dataset).

  * Time logic — For "night snacks at 11pm on Tuesday," the Python time filter correctly surfaces only venues genuinely open at that hour (e.g., izakayas and late-night bistros) and excludes places closed on Tuesdays.
  * Semantic understanding — For abstract requests like "somewhere romantic for a date," it selects ambient bistros and restaurants rather than fast food.
  * Edge cases — For "breakfast at 5am," it honestly states that nothing is open and suggests alternatives instead of inventing a restaurant.

Overall score: 10/10.

Limitations. This is a small, manually scored test rather than a large-scale test, so the numbers are indicative rather than definitive. Budget and semantic ("vibe") matching depend on the LLM's judgment of the retrieved context and are the most variable; adding explicit budget filtering and stronger semantic handling are the next steps for improvement.

## Contributors

Group 9

Member 1: 111ZU1025 徐語歆: data collection and cleaning, presentation slides and posters.

Member 2: 114ZU1005 張亦漩: data collection and cleaning, presentation slides and posters.

Member 3: 114ZU1011 林子庭: data collection and cleaning, presentation slides and poster, RAG implementation, CSV → PDF data pipeline, Hugging Face deployment, and github writing.

Member 4: 114ZU1054 洪瑋婕: data collection and cleaning, presentation slides and poster.

## Acknowledgments

Professor Pien and the Introduction to AI course at National Chengchi University, for guidance and for the suggestion to restructure the data pipeline into per-restaurant documents.
Restaurant information was sourced from public Google Maps listings and customer reviews.

## References

OpenAI — text-embedding-3-small and gpt-4o-mini (https://platform.openai.com/docs)

ChromaDB — open-source vector database (https://www.trychroma.com)

LangChain PyPDFLoader document loader (https://python.langchain.com)

Gradio — web UI framework (https://www.gradio.app)

Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," 2020.
