import logging
import warnings
from models.model_manager import ModelManager
from utils.indexing import load_index, get_relevant_documents
from inference.answer_generation import generate_answer

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

def main():
    try:
        manager = ModelManager.get_instance()
        index = load_index()
        print("\U0001F4A1 Hệ thống sẵn sàng (Nhập 'exit' để thoát)")

        while True:
            query = input("\n❓ Câu hỏi của bạn: ")
            if query.lower() == 'exit':
                break

            relevant_docs = get_relevant_documents(query, index, texts, manager)
            context = " ".join(relevant_docs)
            answer = generate_answer(query, context, manager)
            print(answer)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()