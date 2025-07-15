import json
import os
import shutil

import requests

class DocumentManagerClient:
    def __init__(self, base_url):
        self.base_url = base_url

    # PDF methods
    def upload_pdf(self, file_path, filename=None, metadata=None):
        url = f"{self.base_url}/api/pdfs/"
        if filename:
            shutil.copy(file_path, filename.replace(os.path.basename(file_path), filename))
            file_path = filename.replace(os.path.basename(file_path), filename)
        files = {'file': open(file_path, 'rb')}
        data = {'metadata': json.dumps(metadata)} if metadata else {}
        response = requests.post(url, files=files, data=data)
        files['file'].close()
        response.raise_for_status()
        os.remove(file_path)
        return response.json()

    def list_pdfs(self, name=None, meta_key=None, meta_value=None):
        url = f"{self.base_url}/api/pdfs/"
        params = {}
        if name:
            params['name'] = name
        if meta_key and meta_value:
            params['meta_key'] = meta_key
            params['meta_value'] = meta_value
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def download_pdf(self, pdf_id, save_path):
        url = f"{self.base_url}/api/pdfs/download/{pdf_id}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def get_pdf(self, pdf_id):
        url = f"{self.base_url}/api/pdfs/{pdf_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def delete_pdf(self, pdf_id):
        url = f"{self.base_url}/api/pdfs/{pdf_id}"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()

    # Attachment methods
    def upload_attachment(self, pdf_id, file_path, metadata=None):
        url = f"{self.base_url}/api/attachments/{pdf_id}"
        files = {'file': open(file_path, 'rb')}
        data = {'metadata': json.dumps(metadata)} if metadata else {}
        response = requests.post(url, files=files, data=data)
        files['file'].close()
        response.raise_for_status()
        return response.json()

    def list_attachments(self, pdf_id=None, name=None, meta_key=None, meta_value=None):
        url = f"{self.base_url}/api/attachments/"
        params = {}
        if pdf_id:
            params['pdf_id'] = pdf_id
        if name:
            params['name'] = name
        if meta_key and meta_value:
            params['meta_key'] = meta_key
            params['meta_value'] = meta_value
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def download_attachment(self, attachment_id, save_path):
        url = f"{self.base_url}/api/attachments/download/{attachment_id}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

    def get_attachment(self, attachment_id):
        url = f"{self.base_url}/api/attachments/{attachment_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def delete_attachment(self, attachment_id):
        url = f"{self.base_url}/api/attachments/{attachment_id}"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    client = DocumentManagerClient("http://ulhchathamster:7575")

    print(client.list_pdfs(name="example"))
    """# Upload a PDF
    print("Uploading PDF...")
    pdf = client.upload_pdf("example.pdf", metadata='{"author": "John Doe"}')
    print("Uploaded PDF:", pdf)

    # List PDFs
    print("Listing PDFs...")
    pdfs = client.list_pdfs()
    print(pdfs)

    pdf = pdfs[0]
    # Get PDF details
    print("Getting PDF details...")
    pdf_details = client.get_pdf(pdf['id'])
    print(pdf_details)

    # Download PDF
    print("Downloading PDF...")
    client.download_pdf(pdf['id'], "downloaded_example.pdf")
    print("Downloaded PDF saved as downloaded_example.pdf")

    # Upload an attachment
    print("Uploading attachment...")
    pdf_id = pdf['id']
    attachment = client.upload_attachment(pdf_id, "attachment.pdf", metadata='{"type": "text"}')
    print("Uploaded attachment:", attachment)


    # List attachments
    print("Listing attachments...")
    attachments = client.list_attachments()
    print(attachments)

    attachment = attachments[0]
    # Get attachment details
    print("Getting attachment details...")
    attachment_details = client.get_attachment(attachment['id'])
    print(attachment_details)

    # Download attachment
    print("Downloading attachment...")
    client.download_attachment(attachment['id'], "downloaded_attachment.txt")
    print("Downloaded attachment saved as downloaded_attachment.txt")

    # Delete attachment
    print("Deleting attachment...")
    delete_attachment_response = client.delete_attachment(attachment['id'])
    print(delete_attachment_response)

    # Delete PDF
    print("Deleting PDF...")
    delete_response = client.delete_pdf(pdf['id'])
    print(delete_response)"""