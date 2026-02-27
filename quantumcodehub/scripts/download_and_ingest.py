#!/usr/bin/env python3
"""
Download and ingest quantum computing documentation from official sources
"""

import sys
import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from tqdm import tqdm
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_ingestor import DocumentIngestor
from config.constants import Framework, DocType

class QuantumDocDownloader:
    """Download documentation from quantum frameworks"""
    
    def __init__(self):
        self.ingestor = DocumentIngestor()
        
        # Documentation sources
        self.sources = {
            Framework.QISKIT: {
                'api': 'https://qiskit.org/documentation/apidoc/',
                'tutorials': 'https://qiskit.org/documentation/tutorials.html',
                'release_notes': 'https://qiskit.org/documentation/release_notes.html'
            },
            Framework.PENNYLANE: {
                'api': 'https://docs.pennylane.ai/en/stable/',
                'tutorials': 'https://pennylane.ai/qml/',
                'release_notes': 'https://docs.pennylane.ai/en/stable/development/release_notes.html'
            },
            Framework.CIRQ: {
                'api': 'https://quantumai.google/cirq/api_docs',
                'tutorials': 'https://quantumai.google/cirq/tutorials',
                'release_notes': 'https://quantumai.google/cirq/releases'
            }
        }
    
    def download_page(self, url):
        """Download a single page"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def extract_text_content(self, html, framework, doc_type, version="latest"):
        """Extract meaningful text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def process_framework(self, framework):
        """Process all documentation for a framework"""
        logger.info(f"Processing {framework.value} documentation...")
        
        sources = self.sources.get(framework, {})
        
        for doc_type, url in sources.items():
            logger.info(f"Downloading {doc_type} from {url}")
            
            html = self.download_page(url)
            if not html:
                continue
            
            # Extract main content
            content = self.extract_text_content(html, framework, doc_type)
            
            # Create document
            doc = {
                'content': content,
                'metadata': {
                    'framework': framework.value,
                    'version': 'latest',
                    'doc_type': doc_type,
                    'source_url': url,
                    'source_title': f"{framework.value} {doc_type}",
                    'source_priority': 5 if doc_type == 'api' else 4,
                    'code_weight': 0.8 if doc_type == 'api' else 0.5,
                    'research_weight': 0.3
                }
            }
            
            # Ingest document
            success = self.ingestor.ingest_document(
                content=doc['content'],
                metadata=doc['metadata']
            )
            
            if success:
                logger.success(f"Ingested {framework.value} {doc_type}")
            
            # Be nice to the servers
            time.sleep(2)
    
    def download_sample_data(self):
        """Download and ingest sample data for testing"""
        
        # Create sample documents for each framework
        sample_docs = [
            {
                'content': """
                QuantumCircuit: Create a new quantum circuit.
                
                Example:
                from qiskit import QuantumCircuit
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0,1], [0,1])
                
                Parameters:
                - num_qubits: Number of qubits in the circuit
                - num_classical: Number of classical bits
                - name: Name of the circuit
                """,
                'metadata': {
                    'framework': 'qiskit',
                    'version': '1.2.0',
                    'doc_type': 'api',
                    'is_deprecated': False,
                    'source_priority': 5,
                    'code_weight': 0.9,
                    'source_url': 'https://qiskit.org/documentation/stubs/qiskit.QuantumCircuit.html'
                }
            },
            {
                'content': """
                DEPRECATED: QuantumProgram in qiskit
                The QuantumProgram class is deprecated since qiskit 1.0.0.
                Use QuantumCircuit instead.
                
                Migration example:
                Old: program = QuantumProgram()
                     qr = program.create_quantum_register(2)
                     circuit = program.create_circuit(qr)
                
                New: from qiskit import QuantumCircuit
                     circuit = QuantumCircuit(2)
                """,
                'metadata': {
                    'framework': 'qiskit',
                    'version': '1.0.0',
                    'doc_type': 'deprecation',
                    'is_deprecated': True,
                    'replacement': 'QuantumCircuit',
                    'source_priority': 5,
                    'code_weight': 0.7
                }
            },
            {
                'content': """
                Quantum Fourier Transform Algorithm
                
                The quantum Fourier transform (QFT) is a linear transformation on quantum bits,
                and is the quantum analogue of the discrete Fourier transform.
                
                Complexity: O(n^2) gates for n qubits
                
                Applications:
                - Shor's algorithm
                - Phase estimation
                - Hidden subgroup problems
                
                References:
                - Nielsen & Chuang, Chapter 5
                - arXiv:quant-ph/9608018
                """,
                'metadata': {
                    'framework': 'qiskit',
                    'version': '1.2.0',
                    'doc_type': 'research',
                    'is_deprecated': False,
                    'source_priority': 4,
                    'research_weight': 0.9,
                    'source_url': 'https://arxiv.org/abs/quant-ph/9608018'
                }
            },
            {
                'content': """
                PennyLane: Quantum gradient computation
                
                import pennylane as qml
                
                dev = qml.device('default.qubit', wires=2)
                
                @qml.qnode(dev)
                def circuit(x):
                    qml.RX(x[0], wires=0)
                    qml.RY(x[1], wires=1)
                    qml.CNOT(wires=[0, 1])
                    return qml.expval(qml.PauliZ(0))
                
                # Compute gradient
                x = [0.5, 0.3]
                grad = qml.grad(circuit)(x)
                """,
                'metadata': {
                    'framework': 'pennylane',
                    'version': '0.36.0',
                    'doc_type': 'code_examples',
                    'is_deprecated': False,
                    'source_priority': 4,
                    'code_weight': 0.95,
                    'source_url': 'https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html'
                }
            }
        ]
        
        # Ingest sample documents
        for doc in sample_docs:
            success = self.ingestor.ingest_document(
                content=doc['content'],
                metadata=doc['metadata']
            )
            if success:
                logger.success(f"Ingested sample document: {doc['metadata']['doc_type']}")

def main():
    """Main function to download and ingest documentation"""
    
    downloader = QuantumDocDownloader()
    
    print("=" * 60)
    print("QuantumCodeHub Documentation Downloader & Ingester")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Download sample data only (quick start)")
    print("2. Download from official documentation (may take time)")
    print("3. Download everything (sample + official)")
    print("4. Just initialize empty collections (for testing)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        logger.info("Downloading sample data...")
        downloader.download_sample_data()
        
    elif choice == "2":
        logger.info("Downloading official documentation...")
        for framework in Framework:
            downloader.process_framework(framework)
            
    elif choice == "3":
        logger.info("Downloading everything...")
        downloader.download_sample_data()
        for framework in Framework:
            downloader.process_framework(framework)
            
    elif choice == "4":
        logger.info("Initializing empty collections...")
        from scripts.init_collections import init_empty_collections
        init_empty_collections()
        
    else:
        print("Invalid choice")
        return
    
    # Show stats
    stats = downloader.ingestor.get_stats()
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Total documents ingested: {stats['total_documents']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print("\nCollections:")
    for collection, count in stats['collections'].items():
        print(f"  - {collection}: {count} chunks")

if __name__ == "__main__":
    main()