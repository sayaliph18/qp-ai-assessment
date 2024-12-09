import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFChatbot:
    def __init__(self):
        """Initialize the PDF Chatbot."""
        # Initialize SentenceTransformers model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Compact, fast model for embeddings
        self.index = None  # FAISS index for similarity search
        self.text_chunks = []  # Store text chunks
        self.embeddings = []  # Store embeddings for evaluation

    def upload_pdf(self):
        """Handle PDF file upload."""
        st.header("PDF Chatbot")
        with st.sidebar:
            st.title("Documents")
            uploaded_file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
        return uploaded_file

    def extract_text_from_pdf(self, file):
        """Extract text from a PDF file."""
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    def split_text(self, text):
        """Split the text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        self.text_chunks = text_splitter.split_text(text)
        return self.text_chunks

    def create_vector_store(self):
        """Create FAISS vector store using Sentence Transformers embeddings."""
        # Generate embeddings for all text chunks
        embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]  # Embedding dimensionality

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))  # Add embeddings to the index

    def get_user_question(self):
        """Get the question from the user."""
        user_question = st.text_input("Type your question here:")
        return user_question

    def get_similarity_search_results(self, question, top_k=3):
        """Perform similarity search for the given question."""
        # Embed the user's question
        question_embedding = self.model.encode([question])

        # Perform similarity search using FAISS
        distances, indices = self.index.search(np.array(question_embedding), k=top_k)

        # Return the top-k matching text chunks and their distances
        return [self.text_chunks[i] for i in indices[0]], distances

    def generate_answer(self, context, question):
        """Generate an answer using the context and question."""
        return f"Based on the context: {context}\n\nAnswer to your question: {question}"

    def evaluate_metrics(self, retrieved_chunks, relevant_chunks, top_k=3):
        """Evaluate retrieval performance metrics."""
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_chunks)

        # Precision
        precision = len(relevant_set & retrieved_set) / top_k

        # Recall
        recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0

        return {"Precision": precision, "Recall": recall}

    def run(self):
        """Run the chatbot."""
        file = self.upload_pdf()
        if file is not None:
            # Step 1: Extract text from the uploaded PDF
            text = self.extract_text_from_pdf(file)

            # Step 2: Split text into manageable chunks
            self.split_text(text)

            # Step 3: Create the FAISS vector store
            self.create_vector_store()

            # Step 4: Get user input (question)
            user_question = self.get_user_question()
            if user_question:
                # Step 5: Perform similarity search and generate an answer
                retrieved_chunks, distances = self.get_similarity_search_results(user_question, top_k=3)
                context = "\n".join(retrieved_chunks)  # Concatenate top 3 results as context
                response = self.generate_answer(context, user_question)

                # Step 6: Calculate evaluation metrics
                relevant_chunks = retrieved_chunks  # For simplicity, using retrieved chunks as relevant
                metrics = self.evaluate_metrics(retrieved_chunks, relevant_chunks, top_k=3)

                # Display top 3 results explicitly
                st.write("### Top 3 Retrieved Results:")
                for idx, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Result {idx + 1}:**")
                    st.write(chunk)

                # Display evaluation metrics
                st.write("### Evaluation Metrics:")
                st.write(f"**Precision:** {metrics['Precision']:.2f}")
                st.write(f"**Recall:** {metrics['Recall']:.2f}")



# Create a PDFChatbot instance and run it
if __name__ == "__main__":
    chatbot = PDFChatbot()
    chatbot.run()
