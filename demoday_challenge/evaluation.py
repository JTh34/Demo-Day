# evaluation.py

import logging
import re
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import numpy as np

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)
from ragas import evaluate, RunConfig
from ragas.evaluation import SingleTurnSample

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CanineAppParser:
    """Specialized parser to extract contexts from the canine care application"""
    
    @staticmethod
    def extract_tool_results(message_list):
        """Extracts tool results from a list of messages"""
        context_data = []
        
        # Extract the original question for reference
        original_question = ""
        for msg in message_list:
            if isinstance(msg, HumanMessage):
                original_question = msg.content
                break
                
        logger.debug(f"Original question: {original_question}")
        
        # Process all messages to extract tool results
        for msg in message_list:
            if not isinstance(msg, AIMessage):
                continue
                
            content = msg.content
            
            # Extract RAG tool results
            if "[Using RAG tool]" in content:
                rag_content = content.replace("[Using RAG tool]", "").strip()
                logger.debug(f"Extracted RAG content: {rag_content[:50]}...")
                
                context_data.append({
                    'tool': 'rag',
                    'content': rag_content,
                    'type': 'text'
                })
                
            # Extract Tavily tool results
            elif "[Using Tavily tool]" in content:
                tavily_content = content.replace("[Using Tavily tool]", "").strip()
                logger.debug(f"Raw Tavily content: {tavily_content[:50]}...")
                
                # Parse Internet search results
                results = []
                
                if "Internet search results:" in tavily_content:
                    # Remove header
                    search_results = tavily_content.split("Internet search results:")[1].strip()
                    
                    # Extract numbered results (1., 2., etc.)
                    pattern = r'(\d+)\.\s+(.*?):\s+(.*?)(?=\d+\.|$)'
                    matches = re.findall(pattern, search_results, re.DOTALL)
                    
                    for match in matches:
                        _, title, content = match
                        results.append({
                            'title': title.strip(),
                            'content': content.strip()
                        })
                        
                    logger.debug(f"Number of extracted Tavily results: {len(results)}")
                
                context_data.append({
                    'tool': 'tavily',
                    'results': results,
                    'raw_content': tavily_content,
                    'type': 'search_results'
                })
                
            # Extract final response (without [Using] prefix)
            elif not content.startswith("[Using"):
                # Probably the final response
                context_data.append({
                    'tool': 'final_response',
                    'content': content,
                    'type': 'response'
                })
        
        # Check extraction results
        if not context_data:
            logger.warning(f"No content extracted from messages!")
        else:
            logger.debug(f"Successful extraction: {len(context_data)} context elements")
            
        return context_data
    
    @staticmethod
    def extract_contexts(message_list):
        """Extracts textual contexts from messages for RAGAS evaluation"""
        contexts = []
        
        # Extract all tool results
        tool_results = CanineAppParser.extract_tool_results(message_list)
        
        for result in tool_results:
            # Process RAG results
            if result['tool'] == 'rag':
                # Ignore negative responses
                if not (result['content'].startswith("I'm sorry") and "contains no information" in result['content']):
                    contexts.append(result['content'])
                    logger.debug(f"RAG context added: {result['content'][:50]}...")
            
            # Process Tavily results
            elif result['tool'] == 'tavily':
                if 'results' in result and result['results']:
                    for item in result['results']:
                        if 'content' in item and item['content']:
                            # Ignore empty or too short content
                            if len(item['content']) > 20:
                                contexts.append(item['content'])
                                logger.debug(f"Tavily context added: {item['content'][:50]}...")
                elif 'raw_content' in result:
                    # Fallback if results parsing failed
                    raw_content = result['raw_content']
                    if "Internet search results" in raw_content:
                        cleaned_content = raw_content.replace("Internet search results:", "").strip()
                        contexts.append(cleaned_content)
                        logger.debug(f"Raw Tavily context added: {cleaned_content[:50]}...")
        
        # If no context was found but there is a final response
        if not contexts:
            logger.warning("No context extracted! Check message format.")
            
        return contexts
    
    
    
    @staticmethod
    def extract_final_response(message_list):
        """Extracts the final response from a list of messages"""
        # Find the last AI message that is not a tool result
        for msg in reversed(message_list):
            if isinstance(msg, AIMessage) and not msg.content.startswith("[Using"):
                return msg.content
        
        # Fallback: take the last AI message, whatever it is
        for msg in reversed(message_list):
            if isinstance(msg, AIMessage):
                return msg.content.replace("[Using RAG tool] ", "").replace("[Using Tavily tool] ", "")
        
        return ""

class CanineTestsetGenerator:
    """Test set generator specific to canine care"""
    
    def __init__(self, llm=None, embedding_model=None):
        if llm is None:
            llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        if embedding_model is None:
            embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            
        self.generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
    
    def generate_canine_testset(self, documents, testset_size=10):
        """ Generates a test set specific to canine care  """
        logger.info(f"Generating a canine test dataset with {testset_size} questions...")
        
        # Customize prompt to generate questions about dog care
        custom_prompt = """
        Generate diverse questions about dog and puppy care based on the provided documents.
        Include questions about:
        1. Dog training and behavior
        2. Puppy health and nutrition
        3. Breed-specific care
        4. Managing common puppy problems
        5. Dog safety and first aid
        
        Also include some ambiguous questions or ones that might not have direct answers in the context.
        Include complex questions requiring information to be combined from multiple sections.
        """
        
        # Use standard generator with document context
        try:
            logger.info("TestsetGenerator initialized with old-style parameters")
            
            # Try the traditional generate_with_langchain_docs
            try:
                dataset = self.generator.generate_with_langchain_docs(
                    documents, 
                    testset_size=testset_size
                )
                logger.info(f"Generated {len(dataset)} questions with generate_with_langchain_docs")
            except Exception as e:
                logger.warning(f"generate_with_langchain_docs failed: {str(e)}")
                logger.warning("Trying alternative generation methods...")
                
                # Try generate_from_documents if available
                try:
                    dataset = self.generator.generate_from_documents(
                        documents, 
                        test_size=testset_size
                    )
                    logger.info(f"Generated {len(dataset)} questions with generate_from_documents")
                except Exception as e:
                    logger.warning(f"generate_from_documents failed: {str(e)}")
                    logger.warning("Trying basic generate method...")
                    
                    # Try the basic generate method as a last resort
                    try:
                        dataset = self.generator.generate(
                            docs=documents, 
                            test_size=testset_size
                        )
                        logger.info(f"Generated {len(dataset)} questions with basic generate")
                    except Exception as e:
                        logger.error(f"Error during RAGAS testset generation: {str(e)}")
                        logger.error("Falling back to manual dataset creation if possible or returning empty dataset.")
                        # Return an empty dataset here so we can handle it upstream
                        dataset = EvaluationDataset(samples=[])
        except Exception as e:
            logger.error(f"Unexpected error in test dataset generation: {str(e)}")
            dataset = EvaluationDataset(samples=[])
        
        # Convert the dataset to EvaluationDataset if it's not already one
        if not isinstance(dataset, EvaluationDataset):
            try:
                eval_dataset = dataset.to_evaluation_dataset()
            except Exception as e:
                logger.error(f"Error converting to EvaluationDataset: {str(e)}")
                eval_dataset = EvaluationDataset(samples=[])
        else:
            eval_dataset = dataset

        # Add out-of-domain questions to test detection
        out_of_scope_questions = [
            "What's the weather like in Paris today?",
            "How do I change the oil in my car?",
            "How do I calculate mortgage payments?",
            "Who won the last Super Bowl?",
            "What are the most popular vacation destinations in Europe?",
            "How do I troubleshoot WiFi connection issues?",
            "What is the capital of Australia?",
            "What's the recipe for chocolate chip cookies?"
        ]
        
        # Limit the number of out-of-domain questions to add
        num_out_of_scope = min(testset_size // 4, len(out_of_scope_questions))
        
        logger.info(f"Adding {num_out_of_scope} out-of-domain questions")
        
        # Manually add out-of-domain questions
        for i in range(num_out_of_scope):
            # Create a SingleTurnSample for each out-of-domain question
            # Only use the valid fields supported by SingleTurnSample
            out_of_scope_sample = SingleTurnSample(
                user_input=out_of_scope_questions[i],
                reference="This question is outside the domain of canine care.",
                retrieved_contexts=[]
            )
            
            # Add the sample to the dataset's sample list
            eval_dataset.samples.append(out_of_scope_sample)
            
        logger.info(f"Dataset creation complete. Total questions: {len(eval_dataset.samples)}")
        return eval_dataset

def extract_contexts_for_cohere(message_list):
        """
        Version spéciale pour extraire les contextes de Cohere qui peuvent 
        avoir un format différent
        """
        contexts = []
        
        # Chercher les contextes dans tous les messages
        for msg in message_list:
            if not isinstance(msg, AIMessage):
                continue
                
            content = msg.content
            
            # Chercher les contextes Cohere
            if "reranked results" in content.lower():
                # Essayer d'extraire les résultats reranked
                try:
                    # Format potentiel : "Reranked results: [text1, text2, ...]"
                    start_idx = content.lower().find("reranked results:")
                    if start_idx != -1:
                        results_text = content[start_idx + len("reranked results:"):].strip()
                        # Diviser par lignes ou par un séparateur logique
                        raw_contexts = results_text.split("\n")
                        for ctx in raw_contexts:
                            clean_ctx = ctx.strip()
                            if len(clean_ctx) > 50:  # Longueur minimale
                                contexts.append(clean_ctx)
                                
                except Exception as e:
                    logger.error(f"Error extracting Cohere contexts: {str(e)}")
            
            # Extraire aussi les contextes RAG standards
            if "[Using RAG tool]" in content:
                rag_content = content.replace("[Using RAG tool]", "").strip()
                contexts.append(rag_content)
        
        if not contexts:
            # Chercher n'importe quel contenu substantiel
            for msg in message_list:
                if isinstance(msg, AIMessage) and len(msg.content) > 200:
                    # Prendre les 500 premiers caractères comme contexte
                    contexts.append(msg.content[:500])
                    break
        
        return contexts

class DomainDetectionAccuracy:
    """Measures the accuracy of domain detection (canine vs non-canine questions)"""
    
    name = "domain_detection_accuracy"
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def __call__(self, row):
        """Evaluates if the question is correctly identified as dog-related"""
        question = row["question"]
        response = row["response"]
        
        # Check if the question is about dogs
        dog_question_prompt = f"""
        Is this question about dogs, puppies, or canine care? "{question}"
        
        Examples of dog-related questions:
        - "How do I train my puppy to sit?"
        - "What's the best food for a German Shepherd?"
        - "How do I stop my dog from barking at night?"
        
        Examples of NON-dog-related questions:
        - "What's the weather like in Paris today?"
        - "How do I change the oil in my car?"
        - "What are the best restaurants in New York?"
        
        Answer ONLY with "dog_related" or "not_dog_related".
        """
        
        dog_question_result = self.llm.invoke([SystemMessage(content=dog_question_prompt)])
        is_dog_question = "dog_related" in dog_question_result.content.lower()
        
        # Check if the response is appropriate
        if is_dog_question:
            # For dog questions, the response should NOT contain a domain-based refusal
            domain_appropriate = "I specialize only in canine care" not in response and "outside my area of expertise" not in response
        else:
            # For non-dog questions, the response SHOULD contain a domain-based refusal
            domain_appropriate = "I specialize only in canine care" in response or "outside my area of expertise" in response
        
        logger.info(f"Question: {question[:30]}... | Dog-related: {is_dog_question} | Appropriate response: {domain_appropriate}")
        
        return 1.0 if domain_appropriate else 0.0

def create_manual_eval_dataset():
    """
    Crée un dataset d'évaluation manuel avec des questions prédéfinies
    compatible avec RAGAS sans ajouter d'attributs non autorisés
    """
    # Create dog-related questions
    dog_questions = [
        "How do I house train my puppy?",
        "What's the best way to socialize my puppy with other dogs?",
        "How many meals a day should I feed my puppy?",
        "When can I start training my puppy basic commands?",
        "How do I stop my puppy from biting during play?",
        "What vaccinations does my puppy need?",
        "How long should I walk my puppy each day?",
        "How do I prepare my home for a new puppy?"
    ]
    
    # Create out-of-domain questions
    non_dog_questions = [
        "What's the capital of France?",
        "How do I reset my router?",
        "What's the current stock price of Apple?",
        "Who won the last Super Bowl?"
    ]
    
    # Create samples for each question
    samples = []
    
    # Dog-related questions
    for question in dog_questions:
        # Créer un échantillon avec SEULEMENT les attributs supportés par la classe
        sample = SingleTurnSample(
            user_input=question,
            reference="This is a dog-related question and should be answered with canine expertise.",
            retrieved_contexts=[],
            response=""  # Sera rempli lors de l'évaluation
        )
        
        # NE PAS ajouter d'attributs additionnels ici
        samples.append(sample)
    
    # Non-dog-related questions
    for question in non_dog_questions:
        sample = SingleTurnSample(
            user_input=question,
            reference="This question is outside the domain of canine care.",
            retrieved_contexts=[],
            response=""  # Sera rempli lors de l'évaluation
        )
        
        # NE PAS ajouter d'attributs additionnels ici
        samples.append(sample)
    
    # Create dataset
    dataset = EvaluationDataset(samples=samples)
    
    return dataset

def create_clean_ragas_dataset(samples):
    """
    Crée un dataset RAGAS propre à partir des échantillons
    avec une structure qui préserve à la fois les noms de colonnes originaux
    et les noms standardisés attendus par RAGAS
    """
    clean_data = []
    for sample in samples:
        # Récupérer les valeurs existantes ou définir des valeurs par défaut
        user_input = sample.user_input if hasattr(sample, 'user_input') else ""
        reference = sample.reference if hasattr(sample, 'reference') else ""
        response = sample.response if hasattr(sample, 'response') else ""
        
        # Récupérer les contextes et s'assurer qu'ils ne sont pas vides
        if hasattr(sample, 'retrieved_contexts') and sample.retrieved_contexts:
            if isinstance(sample.retrieved_contexts, list) and sample.retrieved_contexts:
                contexts = [str(ctx) for ctx in sample.retrieved_contexts if ctx is not None and str(ctx).strip()]
            else:
                contexts = []
        else:
            contexts = []
        
        # S'assurer que les contextes ne sont pas vides
        if not contexts:
            contexts = ["This is a placeholder context to ensure metrics can be calculated."]
        
        # Créer un élément propre avec TOUS les noms de colonnes possibles
        # Au lieu de modifier l'objet original, on crée un dictionnaire avec tous les noms
        clean_item = {
            # Noms de colonnes originaux de RAGAS
            'user_input': user_input,
            'reference': reference,
            'retrieved_contexts': contexts,
            'response': response,
            
            # Noms de colonnes standardisés
            'question': user_input,
            'answer': response,
            'contexts': contexts,
            'ground_truth': reference
        }
        
        clean_data.append(clean_item)
    
    # Créer un DataFrame propre et vérifier qu'il contient des données valides
    clean_df = pd.DataFrame(clean_data)
    
    # Journaliser des informations sur le DataFrame
    logger.info(f"Clean DataFrame created with {len(clean_df)} rows")
    logger.info(f"Columns available: {clean_df.columns.tolist()}")
    
    # Vérifier si les contextes sont vides
    if 'retrieved_contexts' in clean_df.columns:
        empty_contexts = clean_df['retrieved_contexts'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
        if empty_contexts > 0:
            logger.warning(f"Warning: {empty_contexts} rows have empty retrieved_contexts")
    
    # Créer un dataset RAGAS
    return EvaluationDataset.from_pandas(clean_df)

def create_evaluation_dataset(documents, num_questions=10):
    """
    Creates an evaluation dataset for testing
    
    Args:
        documents: Documents to use for generating questions
        num_questions: Number of questions to generate
        
    Returns:
        A RAGAS evaluation dataset
    """
    logger.info(f"Creating evaluation dataset with approx. {num_questions} questions...")
    
    # Initialize the dataset generator
    generator = CanineTestsetGenerator()
    
    # Generate the dataset
    dataset = generator.generate_canine_testset(documents, testset_size=num_questions)
    
    # Do NOT try to add custom attributes to SingleTurnSample objects
    # The class is a Pydantic model that validates fields
    
    return dataset

def evaluate_modified_workflow(agent_workflow, dataset, embedding_config=None):
    """
    Version corrigée pour évaluer un workflow avec la structure actuelle du dataset
    
    Args:
        agent_workflow: Le workflow à évaluer
        dataset: Le dataset d'évaluation
        embedding_config: Nom de la configuration d'embedding (optionnel)
        
    Returns:
        Un tuple (standard metrics, domain detection metric)
    """
    logger.info(f"Evaluating workflow with a dataset of {len(dataset)} questions...")
    
    parser = CanineAppParser()
    
    # Traiter chaque question dans le dataset
    for i, sample in enumerate(dataset.samples):
        question = sample.user_input if hasattr(sample, 'user_input') else None
        
        # Vérifier que la question n'est pas None
        if question is None:
            logger.warning(f"Question manquante pour l'échantillon {i+1}, ignoré")
            if hasattr(sample, 'response'):
                sample.response = "ERROR: Question is None"
            if hasattr(sample, 'retrieved_contexts'):
                sample.retrieved_contexts = []
            continue
            
        question_preview = question[:50] if isinstance(question, str) else str(question)
        logger.info(f"Processing question {i+1}/{len(dataset)}: {question_preview}...")
        
        try:
            # Traiter la question avec le workflow
            response = agent_workflow.process_question(question)
            
            # Extraire la réponse finale
            final_response = parser.extract_final_response(response["messages"])
            
            # Extraire les contextes
            if "Cohere" in str(embedding_config):
                contexts = extract_contexts_for_cohere(response["messages"])
            else:
                contexts = parser.extract_contexts(response["messages"])
            contexts = contexts if contexts and isinstance(contexts, list) else []
            
            # Enregistrer la longueur des contextes pour le débogage
            logger.info(f"  - Extracted contexts: {len(contexts)} contexts with total length: {sum(len(c) for c in contexts)} chars")
            
            # Vérifier si les contextes sont vides
            if not contexts or all(not ctx for ctx in contexts):
                # Ajouter un contexte factice pour éviter les métriques à 0
                logger.warning(f"  - No valid contexts found for question {i+1}, adding dummy context")
                dummy_context = "This is a dummy context to ensure metrics can be calculated. " + \
                                "The information in this context is not factual and should not be used for evaluation. " + \
                                "The real context could not be extracted properly."
                contexts = [dummy_context]
            
            # S'assurer que les contextes ne sont pas trop courts
            min_context_length = 50  # Longueur minimale pour un contexte valide
            filtered_contexts = [ctx for ctx in contexts if len(ctx) >= min_context_length]
            if len(filtered_contexts) < len(contexts):
                logger.warning(f"  - Filtered out {len(contexts) - len(filtered_contexts)} too short contexts")
                contexts = filtered_contexts
                if not contexts:
                    logger.warning(f"  - All contexts were too short, adding a dummy context")
                    contexts = ["This is a dummy context with sufficient length to ensure metrics can be calculated properly."]
            
            # Mettre à jour l'échantillon uniquement avec les attributs supportés par la classe
            if hasattr(sample, 'response'):
                sample.response = final_response
            if hasattr(sample, 'retrieved_contexts'):
                sample.retrieved_contexts = contexts
            
            logger.info(f"  - Response length: {len(final_response)}")
            logger.info(f"  - Contexts found: {len(contexts)}")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            error_msg = f"ERROR: {str(e)}"
            if hasattr(sample, 'response'):
                sample.response = error_msg
            if hasattr(sample, 'retrieved_contexts'):
                sample.retrieved_contexts = ["Error occurred during context retrieval"]
    
    # Créer un dataset RAGAS propre avec tous les noms de colonnes possibles
    logger.info("Creating clean RAGAS dataset...")
    evaluation_dataset = create_clean_ragas_dataset(dataset.samples)
    
    # Log des données pour le débogage
    if evaluation_dataset and hasattr(evaluation_dataset, 'to_pandas'):
        df = evaluation_dataset.to_pandas()
        logger.info(f"Dataset structure: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        # Afficher des statistiques sur les longueurs
        if 'retrieved_contexts' in df.columns:
            df['ctx_count'] = df['retrieved_contexts'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            df['ctx_total_len'] = df['retrieved_contexts'].apply(lambda x: sum(len(str(c)) for c in x) if isinstance(x, list) else 0)
            logger.info(f"Context stats: min count={df['ctx_count'].min()}, avg count={df['ctx_count'].mean():.2f}, min total len={df['ctx_total_len'].min()}, avg total len={df['ctx_total_len'].mean():.2f}")
    
    try:
        # Configuration RAGAS avec timeout plus long et moins de workers
        custom_run_config = RunConfig(
            timeout=1200,  # Augmenter le timeout à 20 minutes
            max_workers=2  # Limiter à 2 workers en parallèle
        )
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        
        # Métriques RAGAS standards
        standard_metrics_list = [
            LLMContextRecall(), 
            Faithfulness(),
            ResponseRelevancy(),
            ContextEntityRecall()
        ]
        
        # Mapping de colonnes COMPLET pour RAGAS
        column_map = {
            # Mapping explicite pour toutes les colonnes
            "user_input": "user_input",
            "reference": "reference",
            "response": "response",
            "retrieved_contexts": "retrieved_contexts",
            "question": "user_input",      # Map 'question' TO 'user_input'
            "answer": "response",          # Map 'answer' TO 'response'  
            "contexts": "retrieved_contexts",  # Map 'contexts' TO 'retrieved_contexts'
            "ground_truth": "reference"    # Map 'ground_truth' TO 'reference'
        }
        
        # Lancer l'évaluation RAGAS
        logger.info("Starting RAGAS evaluation...")
        
        try:
            # Essayer l'évaluation avec toutes les métriques
            ragas_result = evaluate(
                dataset=evaluation_dataset,
                metrics=standard_metrics_list,
                llm=evaluator_llm,
                run_config=custom_run_config,
                column_map=column_map
            )
            logger.info("RAGAS evaluation completed successfully with all metrics")
        except Exception as e:
            logger.error(f"Full evaluation failed: {str(e)}")
            logger.info("Trying individual metrics...")
            
            # Essayer chaque métrique individuellement
            ragas_result = {}
            for metric in standard_metrics_list:
                try:
                    logger.info(f"Evaluating with metric: {metric.name}")
                    single_result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=[metric],
                        llm=evaluator_llm,
                        run_config=custom_run_config,
                        column_map=column_map
                    )
                    
                    if hasattr(single_result, 'to_pandas'):
                        df = single_result.to_pandas()
                        if metric.name in df.columns:
                            ragas_result[metric.name] = df[metric.name].mean()
                            logger.info(f"Metric {metric.name} evaluated successfully: {ragas_result[metric.name]}")
                except Exception as inner_e:
                    logger.error(f"Error evaluating metric {metric.name}: {str(inner_e)}")
        
        # Initialiser les métriques standards avec des valeurs par défaut
        standard_metrics = {
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_entity_recall": 0.0
        }
        
        # Mapping des noms de métriques entre RAGAS et notre format
        metric_mapping = {
            "llm_context_recall": "context_recall",
            "faithfulness": "faithfulness",
            "response_relevancy": "answer_relevancy",
            "context_entity_recall": "context_entity_recall"
        }
        
        # Extraire les résultats du format disponible
        if isinstance(ragas_result, dict):
            # Si nous avons utilisé l'évaluation individuelle
            for ragas_name, our_name in metric_mapping.items():
                if ragas_name in ragas_result:
                    standard_metrics[our_name] = round(float(ragas_result[ragas_name]), 2)
                    logger.info(f"Metric {our_name}: {standard_metrics[our_name]}")
        elif hasattr(ragas_result, 'to_pandas'):
            # Si nous avons utilisé l'évaluation complète
            results_df = ragas_result.to_pandas()
            logger.info(f"Result columns: {results_df.columns}")
            
            # Extraire les métriques du DataFrame
            for ragas_name, our_name in metric_mapping.items():
                if ragas_name in results_df.columns:
                    try:
                        standard_metrics[our_name] = round(float(results_df[ragas_name].mean()), 2)
                        logger.info(f"Metric {our_name}: {standard_metrics[our_name]}")
                    except Exception as e:
                        logger.error(f"Error extracting metric {ragas_name}: {str(e)}")
        else:
            # Extraction directe de l'objet result
            for ragas_name, our_name in metric_mapping.items():
                if hasattr(ragas_result, ragas_name):
                    try:
                        value = getattr(ragas_result, ragas_name)
                        if isinstance(value, (int, float)):
                            standard_metrics[our_name] = round(float(value), 2)
                            logger.info(f"Metric {our_name}: {standard_metrics[our_name]}")
                    except Exception as e:
                        logger.error(f"Error accessing metric {ragas_name}: {str(e)}")
        
    except Exception as e:
        # Journalisation détaillée des erreurs
        logger.error(f"Error during RAGAS evaluation: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Métriques par défaut en cas d'échec
        standard_metrics = {
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_entity_recall": 0.0
        }
    
    # Évaluer la détection de domaine
    domain_metric = _evaluate_domain_detection(dataset)
    
    # Arrondir les valeurs à 2 décimales pour l'affichage
    for key in standard_metrics:
        standard_metrics[key] = round(standard_metrics[key], 2)
    
    return standard_metrics, domain_metric

def evaluate_hybrid_workflow(agent_workflow, dataset, embedding_config=None):
    """
    Version spécialisée pour évaluer un workflow hybride avec différentes sources de contexte
    
    Args:
        agent_workflow: Le workflow à évaluer
        dataset: Le dataset d'évaluation
        embedding_config: Nom de la configuration d'embedding (optionnel)
        
    Returns:
        Un tuple (metrics_by_category, overall_metrics, domain_metric)
    """
    logger.info(f"Evaluating hybrid workflow with a dataset of {len(dataset)} questions...")
    
    parser = CanineAppParser()
    
    # Catégories pour analyser les résultats séparément
    categories = {
        'rag': {'count': 0, 'samples': []},
        'tavily': {'count': 0, 'samples': []},
        'rejected': {'count': 0, 'samples': []}
    }
    
    # Traiter chaque question dans le dataset
    for i, sample in enumerate(dataset.samples):
        question = sample.user_input if hasattr(sample, 'user_input') else None
        
        # Vérifier que la question n'est pas None
        if question is None:
            logger.warning(f"Question manquante pour l'échantillon {i+1}, ignoré")
            continue
            
        question_preview = question[:50] if isinstance(question, str) else str(question)
        logger.info(f"Processing question {i+1}/{len(dataset)}: {question_preview}...")
        
        try:
            # Traiter la question avec le workflow
            response = agent_workflow.process_question(question)
            
            # Extraire la réponse finale
            final_response = parser.extract_final_response(response["messages"])
            
            # Extraire les contextes et déterminer le type de réponse
            rag_contexts = []
            tavily_contexts = []
            is_rejected = False
            
            # Analyser les messages pour détecter le type de réponse
            for msg in response["messages"]:
                if not isinstance(msg, AIMessage):
                    continue
                
                content = msg.content
                
                # Vérifier si c'est une réponse via RAG
                if "[Using RAG tool]" in content:
                    rag_contexts = parser.extract_contexts([msg])
                    
                # Vérifier si c'est une réponse via Tavily
                elif "[Using Tavily tool]" in content:
                    tavily_contexts = parser.extract_contexts([msg])
                
                # Vérifier si c'est une réponse de rejet (hors sujet)
                elif "I specialize only in canine care" in content or "outside my area of expertise" in content:
                    is_rejected = True
            
            # Déterminer la catégorie principale de la réponse
            if is_rejected:
                category = 'rejected'
                contexts = []  # Aucun contexte pour les réponses rejetées
            elif rag_contexts:
                category = 'rag'
                contexts = rag_contexts
            elif tavily_contexts:
                category = 'tavily'
                contexts = tavily_contexts
            else:
                category = 'rag'  # Par défaut, considérer comme RAG si indéterminé
                contexts = parser.extract_contexts(response["messages"])
            
            # Créer une copie de l'échantillon pour la catégorisation
            import copy
            sample_copy = copy.deepcopy(sample)
            
            # Mettre à jour la copie avec la réponse et les contextes
            # Ne modifier que les attributs standards supportés par SingleTurnSample
            if hasattr(sample_copy, 'response'):
                sample_copy.response = final_response
            if hasattr(sample_copy, 'retrieved_contexts'):
                sample_copy.retrieved_contexts = contexts
            
            # Mettre à jour l'échantillon original aussi (pour la rétrocompatibilité)
            sample.response = final_response
            sample.retrieved_contexts = contexts
            
            # Au lieu d'ajouter un attribut non supporté, utiliser une variable séparée
            # pour tracer la catégorie (lors de la création du dataframe plus tard)
            current_category = category
            
            # Ajouter à la catégorie appropriée
            categories[category]['count'] += 1
            categories[category]['samples'].append(sample_copy)
            
            logger.info(f"  - Question categorized as: {category.upper()}")
            logger.info(f"  - Response length: {len(final_response)}")
            logger.info(f"  - Contexts found: {len(contexts)}")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            error_msg = f"ERROR: {str(e)}"
            
            # En cas d'erreur, on ne catégorise pas l'échantillon
            sample.response = error_msg
            sample.retrieved_contexts = ["Error processing question"]
    
    # Afficher les statistiques de catégorisation
    logger.info(f"Category statistics:")
    logger.info(f"  - RAG responses: {categories['rag']['count']}/{len(dataset)} ({categories['rag']['count']/len(dataset)*100:.1f}%)")
    logger.info(f"  - Tavily responses: {categories['tavily']['count']}/{len(dataset)} ({categories['tavily']['count']/len(dataset)*100:.1f}%)")
    logger.info(f"  - Rejected questions: {categories['rejected']['count']}/{len(dataset)} ({categories['rejected']['count']/len(dataset)*100:.1f}%)")
    
    # Évaluer chaque catégorie séparément
    metrics_by_category = {}
    
    # Évaluer uniquement les catégories avec des échantillons
    for category, data in categories.items():
        if data['count'] > 0:
            logger.info(f"Evaluating {category.upper()} category ({data['count']} samples)...")
            
            # Créer un dataset pour cette catégorie
            category_dataset = EvaluationDataset(samples=data['samples'])
            
            # Adapter l'évaluation selon la catégorie
            if category == 'rag':
                # Pour les réponses RAG, toutes les métriques sont pertinentes
                try:
                    metrics = evaluate_ragas_metrics(category_dataset, 
                                                    ["context_recall", "faithfulness", "answer_relevancy", "context_entity_recall"])
                    metrics_by_category[category] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating RAG category: {str(e)}")
                    metrics_by_category[category] = {
                        "context_recall": 0.0, 
                        "faithfulness": 0.0, 
                        "answer_relevancy": 0.0, 
                        "context_entity_recall": 0.0
                    }
            
            elif category == 'tavily':
                # Pour les réponses Tavily, certaines métriques sont pertinentes
                try:
                    metrics = evaluate_ragas_metrics(category_dataset, 
                                                    ["faithfulness", "answer_relevancy"])
                    metrics_by_category[category] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating Tavily category: {str(e)}")
                    metrics_by_category[category] = {
                        "faithfulness": 0.0, 
                        "answer_relevancy": 0.0
                    }
            
            elif category == 'rejected':
                # Pour les questions rejetées, seule la pertinence de la réponse est pertinente
                try:
                    metrics = evaluate_ragas_metrics(category_dataset, 
                                                    ["answer_relevancy"])
                    metrics_by_category[category] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating Rejected category: {str(e)}")
                    metrics_by_category[category] = {
                        "answer_relevancy": 0.0
                    }
    
    # Évaluer sur l'ensemble du dataset pour la rétrocompatibilité
    try:
        overall_metrics, domain_metric = evaluate_modified_workflow(agent_workflow, dataset, embedding_config)
    except Exception as e:
        logger.error(f"Error evaluating overall dataset: {str(e)}")
        overall_metrics = {
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_entity_recall": 0.0
        }
        domain_metric = _evaluate_domain_detection(dataset)
    
    return metrics_by_category, overall_metrics, domain_metric

def evaluate_ragas_metrics(dataset, metric_names=None):
    """
    Évalue un dataset RAGAS avec des métriques spécifiques
    
    Args:
        dataset: Dataset d'évaluation RAGAS
        metric_names: Liste des noms de métriques à évaluer
        
    Returns:
        Dictionnaire des résultats de métriques
    """
    if not metric_names:
        metric_names = ["context_recall", "faithfulness", "answer_relevancy", "context_entity_recall"]
    
    # Mapping des noms de métriques aux classes de métriques
    metric_mapping = {
        "context_recall": LLMContextRecall,
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "context_entity_recall": ContextEntityRecall
    }
    
    # Créer un dataset propre
    clean_dataset = create_clean_ragas_dataset(dataset.samples)
    
    # Initialiser les résultats
    results = {}
    
    # Configuration de l'évaluation
    custom_run_config = RunConfig(
        timeout=600,  # 10 minutes par métrique
        max_workers=2
    )
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    
    # Mapping de colonnes complet
    column_map = {
        "user_input": "user_input",
        "reference": "reference",
        "response": "response",
        "retrieved_contexts": "retrieved_contexts",
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    }
    
    # Évaluer chaque métrique séparément
    for metric_name in metric_names:
        if metric_name in metric_mapping:
            try:
                logger.info(f"Evaluating metric: {metric_name}")
                
                # Créer la métrique
                metric = metric_mapping[metric_name]()
                
                # Évaluer avec cette métrique uniquement
                result = evaluate(
                    dataset=clean_dataset,
                    metrics=[metric],
                    llm=evaluator_llm,
                    run_config=custom_run_config,
                    column_map=column_map
                )
                
                # Extraire le résultat
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    ragas_name = metric.name
                    if ragas_name in df.columns:
                        value = float(df[ragas_name].mean())
                        results[metric_name] = round(value, 2)
                        logger.info(f"  - {metric_name}: {results[metric_name]}")
            except Exception as e:
                logger.error(f"Error evaluating {metric_name}: {str(e)}")
                results[metric_name] = 0.0
    
    return results

def _evaluate_domain_detection(dataset):
    """Specifically evaluates domain detection"""
    logger.info("Evaluating domain detection accuracy...")
    
    domain_metric = DomainDetectionAccuracy()
    scores = []
    
    for i, sample in enumerate(dataset.samples):
        # Utiliser 'user_input' ou 'question' selon ce qui est disponible
        question = None
        if hasattr(sample, 'user_input'):
            question = sample.user_input
        elif hasattr(sample, 'question'):
            question = sample.question
            
        # Utiliser 'response' ou 'answer' selon ce qui est disponible
        response = None
        if hasattr(sample, 'response'):
            response = sample.response
        elif hasattr(sample, 'answer'):
            response = sample.answer
            
        # Vérifier que question et response ne sont pas None
        if question is None or response is None:
            logger.warning(f"Question ou réponse manquante pour l'échantillon {i+1}, ignoré")
            continue
            
        # Utiliser un slicing sécurisé
        question_preview = question[:50] if isinstance(question, str) else str(question)
        logger.info(f"Evaluating domain detection for question {i+1}: {question_preview}...")
        
        try:
            score = domain_metric({
                "question": question,
                "response": response
            })
            scores.append(score)
            
            logger.info(f"  - Domain detection score: {score}")
        except Exception as e:
            logger.error(f"Error evaluating domain detection: {str(e)}")
    
    # Calculate average
    avg_score = sum(scores) / len(scores) if scores else 0
    logger.info(f"Overall domain detection accuracy: {avg_score:.4f}")
    
    return avg_score

def evaluate_canine_application(agent_workflow, documents, num_questions=10, embedding_config=None):
    """ 
    Main function to evaluate the canine care application with RAGAS
    This function combines both steps: dataset creation and evaluation
    """
    logger.info(f"Starting evaluation with {num_questions} test questions...")
    
    # Step 1: Create the dataset
    dataset = create_evaluation_dataset(documents, num_questions)
    
    # Check if dataset is empty or has too few questions
    if len(dataset.samples) < 2:
        logger.warning("Dataset creation failed or returned too few questions, using manual dataset")
        dataset = create_manual_eval_dataset()
        logger.info(f"Created manual dataset with {len(dataset.samples)} questions")
    
    # Step 2: Evaluate the workflow with the dataset
    try:
        metrics_by_category, overall_metrics, domain_metric = evaluate_hybrid_workflow(
            agent_workflow, 
            dataset,
            embedding_config
        )
        
        # Return a tuple with all metrics for full analysis
        return metrics_by_category, overall_metrics, domain_metric, dataset
    except Exception as e:
        logger.error(f"Hybrid evaluation failed, falling back to standard evaluation: {str(e)}")
        
        # Fallback to standard evaluation
        standard_metrics, domain_metric = evaluate_modified_workflow(
            agent_workflow, 
            dataset,
            embedding_config
        )
        
        # Return a compatible tuple with standard metrics
        return {}, standard_metrics, domain_metric, dataset

def compare_embedding_models(agent_workflows, documents, num_questions=5):
    """ Compare different embedding configurations """
    logger.info(f"Comparing {len(agent_workflows)} embedding configurations...")
    
    # Subsample documents to speed up evaluation
    sample_size = min(50, len(documents))
    sample_documents = random.sample(documents, sample_size)
    logger.info(f"Using a sample of {sample_size} documents for evaluation")
    
    # Step 1: Create a common dataset for all configurations
    logger.info("Generating a common test dataset...")
    dataset = create_evaluation_dataset(sample_documents, num_questions)
    
    # Check if dataset creation failed
    if len(dataset.samples) < 2:
        logger.warning("Dataset creation failed, using manual dataset")
        dataset = create_manual_eval_dataset()
    
    logger.info(f"Dataset generated with {len(dataset.samples)} questions")
    
    # Store results for each configuration
    all_results = {}
    
    # Step 2: Evaluate each configuration with the same dataset
    for name, workflow in agent_workflows.items():
        logger.info(f"Evaluating configuration: {name}")
        
        try:
            # Try to use the hybrid evaluation
            metrics_by_category, standard_metrics, domain_metric = evaluate_hybrid_workflow(
                workflow, 
                dataset,
                embedding_config=name
            )
            
            # Combine category metrics if available
            if metrics_by_category:
                category_average = {}
                all_metrics = set()
                
                # Collect all metrics from all categories
                for category, metrics in metrics_by_category.items():
                    for metric_name in metrics.keys():
                        all_metrics.add(metric_name)
                
                # Average metrics across categories
                for metric_name in all_metrics:
                    values = []
                    for category, metrics in metrics_by_category.items():
                        if metric_name in metrics:
                            values.append(metrics[metric_name])
                    
                    if values:
                        category_average[f"{metric_name}_avg"] = sum(values) / len(values)
                
                # Combine metrics
                all_results[name] = {
                    **standard_metrics,
                    **category_average,
                    "domain_detection": domain_metric
                }
            else:
                # Use standard metrics if categories aren't available
                all_results[name] = {
                    **standard_metrics,
                    "domain_detection": domain_metric
                }
                
        except Exception as e:
            logger.error(f"Hybrid evaluation failed for {name}: {str(e)}")
            logger.info(f"Falling back to standard evaluation...")
            
            # Fallback to standard evaluation
            standard_metrics, domain_metric = evaluate_modified_workflow(
                workflow, 
                dataset,
                embedding_config=name
            )
            
            all_results[name] = {
                **standard_metrics,
                "domain_detection": domain_metric
            }
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results).T
    
    # Save results
    results_df.to_csv("embedding_comparison_results.csv")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Embedding Configuration Comparison')
    plt.ylabel('Score')
    plt.xlabel('Configuration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('embedding_comparison_results.png')
    
    logger.info("Comparison completed.")
    
    return results_df