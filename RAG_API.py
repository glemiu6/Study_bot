from abc import ABC,abstractmethod
class RAG(ABC):
    @abstractmethod
    def read_files(self,file):
        """
        :param file: file
        :return: text from file
        """
        pass
    @abstractmethod
    def _get_collection_create(self):
        """

        :return: collection from chroma
        """
        pass

    @abstractmethod
    def _hash_text(self,text):
        """
        encode the string
        :param text: string
        :return: hexadecimal code
        """
        pass

    @abstractmethod
    def _read_pdf(self,file):
        """requirements: file pdf"""
        pass

    @abstractmethod
    def _read_docx(self,file):
        """requirements: file docx"""
        pass

    @abstractmethod
    def _read_md(self,file):
        """requirements: file md"""
        pass

    @abstractmethod
    def _chunk_text(self,text,chunk_size=500,chunk_overlap=100):
        """

        :param text: string from file
        :param chunk_size: size os the text extracted
        :param chunk_overlap: smoother text
        :return: parts of the original file
        """
        pass

    @abstractmethod
    def transformer(self,chunk):
        """
        requirements: using _chunk_text()
        :param chunk: parts of the file after chunking
        :return: vectors
        """
        pass

    @abstractmethod
    def vector_db(self,chunk,transform,file):
        """

        :param chunk:
        :param transform:
        :param file:

        """
        pass


    @abstractmethod
    def process(self,file):
        """

        :param file: string
        :return: store in vectors
        """
        pass

    @abstractmethod
    def query(self,file,question,top_k=3,chat_history=None):
        """

        :param file: text
        :param question: string
        :param top_k: the top k most similar to the question
        :param chat_history: history
        :return: respons from the database
        """
        pass