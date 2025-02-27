import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class RelationUpdateLayer(nn.Module):
    """
    Update layer for relations in the RRN model.
    Implements the update operations for triples involving relations.
    """
    def __init__(self, embedding_dim):
        super(RelationUpdateLayer, self).__init__()
        # Parameters for gate computation
        self.V1 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.V2 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        
        # Parameters for candidate update computation
        self.W1 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W2 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.w = nn.Parameter(torch.randn(embedding_dim))
        
    def forward(self, e_s, e_o):
        """
        Update individual embeddings based on relations.
        
        Args:
            e_s: Subject embedding
            e_o: Object embedding
            
        Returns:
            Updated subject embedding
        """
        # Gate computation (determines how much to update)
        g = torch.sigmoid(torch.matmul(e_s, self.V1) + torch.matmul(e_o, self.V2))
        
        # Candidate update computation
        e_tilde = F.relu(
            torch.matmul(e_s, self.W1) + 
            torch.matmul(e_o, self.W2) + 
            torch.sum(e_s.unsqueeze(1) * e_o.unsqueeze(0) * self.w, dim=1)
        )
        
        # Apply gate to candidate update - create a new tensor rather than modifying in-place
        e_updated = e_s + e_tilde * g
        
        # Normalize to maintain unit vectors - create a new tensor with F.normalize
        # This avoids in-place modification that breaks autograd
        e_updated = F.normalize(e_updated, p=2, dim=0)
        
        return e_updated

class ClassUpdateLayer(nn.Module):
    """
    Update layer for class memberships in the RRN model.
    Implements the update operations for triples involving class memberships.
    """
    def __init__(self, embedding_dim, num_classes):
        super(ClassUpdateLayer, self).__init__()
        # Parameters for gate and candidate update computation
        self.V = nn.Parameter(torch.randn(embedding_dim + num_classes, embedding_dim))
        self.W = nn.Parameter(torch.randn(embedding_dim + num_classes, embedding_dim))
        
    def forward(self, e_i, class_indicator):
        """
        Update individual embeddings based on class memberships.
        
        Args:
            e_i: Individual embedding
            class_indicator: Binary vector indicating class memberships
            
        Returns:
            Updated individual embedding
        """
        # Concatenate embedding with class indicator
        concat = torch.cat([e_i, class_indicator])
        
        # Gate computation
        g = torch.sigmoid(torch.matmul(self.V.t(), concat))
        
        # Candidate update computation
        e_tilde = F.relu(torch.matmul(self.W.t(), concat))
        
        # Apply gate to candidate update - create a new tensor
        e_updated = e_i + e_tilde * g
        
        # Normalize to maintain unit vectors - avoid in-place operations
        e_updated = F.normalize(e_updated, p=2, dim=0)
        
        return e_updated

class ClassPredictionMLP(nn.Module):
    """
    Multi-layer perceptron for predicting class memberships.
    """
    def __init__(self, embedding_dim, num_classes):
        super(ClassPredictionMLP, self).__init__()
        hidden_dim = embedding_dim  # Hidden layer size
        
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, e_i):
        """
        Predict class memberships for an individual.
        
        Args:
            e_i: Individual embedding
            
        Returns:
            Probabilities of class memberships
        """
        h = F.relu(self.hidden(e_i))
        return torch.sigmoid(self.output(h))

class RelationPredictionMLP(nn.Module):
    """
    Multi-layer perceptron for predicting relations between individuals.
    """
    def __init__(self, embedding_dim):
        super(RelationPredictionMLP, self).__init__()
        hidden_dim = embedding_dim  # Hidden layer size
        
        self.hidden = nn.Linear(2 * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, e_s, e_o):
        """
        Predict relation between two individuals.
        
        Args:
            e_s: Subject embedding
            e_o: Object embedding
            
        Returns:
            Probability of relation
        """
        # Concatenate subject and object embeddings
        concat = torch.cat([e_s, e_o])
        h = F.relu(self.hidden(concat))
        return torch.sigmoid(self.output(h).squeeze())

class RRN(nn.Module):
    """
    Recursive Reasoning Network (RRN) for ontology reasoning.
    """
    def __init__(self, embedding_dim, num_classes, num_relations):
        super(RRN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_relations = num_relations
        
        # Class update layer
        self.class_update = ClassUpdateLayer(embedding_dim, num_classes)
        
        # Relation update layers (one for subjects and one for objects per relation)
        self.subject_update_layers = nn.ModuleList(
            [RelationUpdateLayer(embedding_dim) for _ in range(num_relations)]
        )
        self.object_update_layers = nn.ModuleList(
            [RelationUpdateLayer(embedding_dim) for _ in range(num_relations)]
        )
        
        # Class prediction layer
        self.class_predictor = ClassPredictionMLP(embedding_dim, num_classes)
        
        # Relation prediction layers (one per relation)
        self.relation_predictors = nn.ModuleList(
            [RelationPredictionMLP(embedding_dim) for _ in range(num_relations)]
        )
    
    def initialize_embeddings(self, num_individuals):
        """
        Initialize random embeddings for individuals.
        
        Args:
            num_individuals: Number of individuals
            
        Returns:
            Tensor of normalized embeddings
        """
        # Random initialization
        embeddings = torch.randn(num_individuals, self.embedding_dim)
        # Normalize to unit vectors - create a new tensor
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Ensure the tensor requires gradients
        embeddings.requires_grad_(True)
        return embeddings
    
    def generate_embeddings(self, facts, num_individuals, class_memberships, num_iterations):
        """
        Generate embeddings for individuals by iteratively updating based on facts.
        
        Args:
            facts: List of relation facts (subject_idx, relation_idx, object_idx)
            num_individuals: Number of individuals
            class_memberships: List of class memberships (individual_idx, class_idx, value)
            num_iterations: Number of update iterations
            
        Returns:
            Tensor of updated embeddings
        """
        # Initialize embeddings - ensure they require gradients
        embeddings = self.initialize_embeddings(num_individuals)
        
        # Create class indicators
        class_indicators = torch.zeros(num_individuals, self.num_classes)
        for ind_idx, class_idx, value in class_memberships:
            class_indicators[ind_idx, class_idx] = value
        
        # Create a copy of embeddings to avoid in-place operations
        # This is crucial for the gradient computation
        embeddings = embeddings.clone().requires_grad_(True)
        
        for _ in range(num_iterations):
            # Create a new tensor to store updated embeddings
            new_embeddings = embeddings.clone()
            
            # Update based on class memberships
            for ind_idx in range(num_individuals):
                if torch.any(class_indicators[ind_idx] != 0):
                    new_embeddings[ind_idx] = self.class_update(
                        embeddings[ind_idx], class_indicators[ind_idx]
                    )
            
            # Update based on relations - create a new tensor for each update
            for subj_idx, rel_idx, obj_idx in facts:
                # Update subject embedding
                new_embeddings[subj_idx] = self.subject_update_layers[rel_idx](
                    embeddings[subj_idx], embeddings[obj_idx]
                )
                
                # Update object embedding
                new_embeddings[obj_idx] = self.object_update_layers[rel_idx](
                    embeddings[obj_idx], embeddings[subj_idx]
                )
            
            # Replace the old embeddings with the new ones
            embeddings = new_embeddings
                
        return embeddings
    
    def forward(self, kb, num_iterations=5):
        """
        Forward pass for the RRN model.
        
        Args:
            kb: Knowledge base with facts, class memberships, and queries
            num_iterations: Number of update iterations
            
        Returns:
            Tensor of predictions for queries
        """
        facts = kb['facts']
        num_individuals = kb['num_individuals']
        class_memberships = kb['class_memberships']
        queries = kb['queries']
        
        # Generate embeddings based on facts
        embeddings = self.generate_embeddings(
            facts, num_individuals, class_memberships, num_iterations
        )
        
        # Make predictions for queries
        predictions = []
        for query_type, *query_params in queries:
            if query_type == 'class':
                ind_idx, class_idx = query_params
                class_preds = self.class_predictor(embeddings[ind_idx])
                predictions.append(class_preds[class_idx])
            elif query_type == 'relation':
                subj_idx, rel_idx, obj_idx = query_params
                rel_pred = self.relation_predictors[rel_idx](
                    embeddings[subj_idx], embeddings[obj_idx]
                )
                predictions.append(rel_pred)
                
        return torch.stack(predictions) if predictions else torch.tensor([])

def preprocess_kb(kb, entity_to_idx, class_to_idx, relation_to_idx):
    """
    Preprocess a knowledge base for the RRN model.
    
    Args:
        kb: Raw knowledge base with facts and queries
        entity_to_idx: Mapping from entity names to indices
        class_to_idx: Mapping from class names to indices
        relation_to_idx: Mapping from relation names to indices
        
    Returns:
        Processed knowledge base with indices
    """
    processed_kb = {
        'num_individuals': len(entity_to_idx),
        'facts': [],
        'class_memberships': [],
        'queries': [],
        'targets': []
    }
    
    # Process facts
    for fact in kb['facts']:
        if fact['type'] == 'relation':
            subj_idx = entity_to_idx[fact['subject']]
            rel_idx = relation_to_idx[fact['relation']]
            obj_idx = entity_to_idx[fact['object']]
            processed_kb['facts'].append((subj_idx, rel_idx, obj_idx))
        elif fact['type'] == 'class':
            ind_idx = entity_to_idx[fact['individual']]
            class_idx = class_to_idx[fact['class']]
            value = 1 if fact.get('value', True) else -1
            processed_kb['class_memberships'].append((ind_idx, class_idx, value))
    
    # Process queries
    for query in kb['queries']:
        if query['type'] == 'relation':
            subj_idx = entity_to_idx[query['subject']]
            rel_idx = relation_to_idx[query['relation']]
            obj_idx = entity_to_idx[query['object']]
            processed_kb['queries'].append(('relation', subj_idx, rel_idx, obj_idx))
            processed_kb['targets'].append(1.0 if query.get('value', True) else 0.0)
        elif query['type'] == 'class':
            ind_idx = entity_to_idx[query['individual']]
            class_idx = class_to_idx[query['class']]
            processed_kb['queries'].append(('class', ind_idx, class_idx))
            processed_kb['targets'].append(1.0 if query.get('value', True) else 0.0)
    
    processed_kb['targets'] = torch.tensor(processed_kb['targets'])
    return processed_kb

def load_data(data_path):
    """
    Load and preprocess data from a JSON file.
    
    Args:
        data_path: Path to JSON data file
        
    Returns:
        List of processed knowledge bases
        Dictionary mapping entity names to indices
        Dictionary mapping class names to indices
        Dictionary mapping relation names to indices
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create mappings from entity/class/relation names to indices
    entity_to_idx = {}
    class_to_idx = {}
    relation_to_idx = {}
    
    # First pass to collect all entities, classes, and relations
    for kb in data:
        for fact in kb['facts']:
            if fact['type'] == 'relation':
                if fact['subject'] not in entity_to_idx:
                    entity_to_idx[fact['subject']] = len(entity_to_idx)
                if fact['object'] not in entity_to_idx:
                    entity_to_idx[fact['object']] = len(entity_to_idx)
                if fact['relation'] not in relation_to_idx:
                    relation_to_idx[fact['relation']] = len(relation_to_idx)
            elif fact['type'] == 'class':
                if fact['individual'] not in entity_to_idx:
                    entity_to_idx[fact['individual']] = len(entity_to_idx)
                if fact['class'] not in class_to_idx:
                    class_to_idx[fact['class']] = len(class_to_idx)
        
        for query in kb['queries']:
            if query['type'] == 'relation':
                if query['subject'] not in entity_to_idx:
                    entity_to_idx[query['subject']] = len(entity_to_idx)
                if query['object'] not in entity_to_idx:
                    entity_to_idx[query['object']] = len(entity_to_idx)
                if query['relation'] not in relation_to_idx:
                    relation_to_idx[query['relation']] = len(relation_to_idx)
            elif query['type'] == 'class':
                if query['individual'] not in entity_to_idx:
                    entity_to_idx[query['individual']] = len(entity_to_idx)
                if query['class'] not in class_to_idx:
                    class_to_idx[query['class']] = len(class_to_idx)
    
    # Second pass to preprocess knowledge bases
    processed_data = [
        preprocess_kb(kb, entity_to_idx, class_to_idx, relation_to_idx)
        for kb in data
    ]
    
    return processed_data, entity_to_idx, class_to_idx, relation_to_idx

def train_rrn(model, train_data, val_data=None, num_epochs=10, lr=0.001, weight_decay=1e-6, log_dir=None):
    """
    Train the RRN model.
    
    Args:
        model: RRN model
        train_data: Training data
        val_data: Validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        log_dir: Directory for TensorBoard logs
        
    Returns:
        Trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Set up TensorBoard logging if specified
    writer = SummaryWriter(log_dir) if log_dir else None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for kb_idx, kb in enumerate(tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Skip empty queries
            if len(kb['queries']) == 0:
                continue
                
            # Forward pass
            predictions = model(kb)
            targets = kb['targets']
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Log training loss to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Evaluate on validation data if provided
        if val_data:
            val_acc = evaluate_rrn(model, val_data)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Log validation accuracy to TensorBoard
            if writer:
                writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    if writer:
        writer.close()
    
    return model

def evaluate_rrn(model, eval_data):
    """
    Evaluate the RRN model.
    
    Args:
        model: RRN model
        eval_data: Evaluation data
        
    Returns:
        Accuracy of the model on the evaluation data
    """
    model.eval()
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for kb in eval_data:
            # Skip empty queries
            if len(kb['queries']) == 0:
                continue
                
            # Forward pass
            predictions = model(kb)
            targets = kb['targets']
            
            # Convert to binary predictions
            binary_preds = (predictions >= 0.5).float()
            
            # Count correct predictions
            correct = (binary_preds == targets).sum().item()
            total_correct += correct
            total_examples += len(targets)
    
    # Avoid division by zero
    if total_examples == 0:
        return 0.0
        
    accuracy = total_correct / total_examples
    return accuracy

def generate_synthetic_data(num_samples=100, num_entities=20, num_classes=3, num_relations=5, num_facts_per_kb=15, num_queries_per_kb=10):
    """
    Generate synthetic data for testing the RRN implementation.
    
    Args:
        num_samples: Number of knowledge bases to generate
        num_entities: Number of entities per knowledge base
        num_classes: Number of classes in the ontology
        num_relations: Number of relations in the ontology
        num_facts_per_kb: Number of facts per knowledge base
        num_queries_per_kb: Number of queries per knowledge base
        
    Returns:
        List of synthetic knowledge bases
    """
    np.random.seed(42)
    
    # Generate entity, class, and relation names
    entities = [f"entity_{i}" for i in range(num_entities)]
    classes = [f"class_{i}" for i in range(num_classes)]
    relations = [f"relation_{i}" for i in range(num_relations)]
    
    data = []
    
    for _ in range(num_samples):
        kb = {
            "facts": [],
            "queries": []
        }
        
        # Generate facts
        for _ in range(num_facts_per_kb):
            fact_type = np.random.choice(["relation", "class"])
            
            if fact_type == "relation":
                subject = np.random.choice(entities)
                relation = np.random.choice(relations)
                object_ = np.random.choice(entities)
                
                kb["facts"].append({
                    "type": "relation",
                    "subject": subject,
                    "relation": relation,
                    "object": object_
                })
            else:  # class fact
                individual = np.random.choice(entities)
                class_ = np.random.choice(classes)
                value = bool(np.random.choice([True, False]))
                
                kb["facts"].append({
                    "type": "class",
                    "individual": individual,
                    "class": class_,
                    "value": value
                })
        
        # Generate queries
        for _ in range(num_queries_per_kb):
            query_type = np.random.choice(["relation", "class"])
            
            if query_type == "relation":
                subject = np.random.choice(entities)
                relation = np.random.choice(relations)
                object_ = np.random.choice(entities)
                value = bool(np.random.choice([True, False]))
                
                kb["queries"].append({
                    "type": "relation",
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "value": value
                })
            else:  # class query
                individual = np.random.choice(entities)
                class_ = np.random.choice(classes)
                value = bool(np.random.choice([True, False]))
                
                kb["queries"].append({
                    "type": "class",
                    "individual": individual,
                    "class": class_,
                    "value": value
                })
        
        data.append(kb)
    
    return data

def main():
    """
    Main function to run the RRN model.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate RRN model')
    parser.add_argument('--data_path', type=str, help='Path to data JSON file')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--output_path', type=str, default='synthetic_data.json', help='Path to save synthetic data')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of embeddings')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of update iterations')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for TensorBoard logs')
    args = parser.parse_args()
    
    # Generate synthetic data if requested
    if args.generate_data:
        print("Generating synthetic data...")
        synthetic_data = generate_synthetic_data()
        
        with open(args.output_path, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
            
        print(f"Synthetic data saved to {args.output_path}")
        
        if not args.data_path:
            args.data_path = args.output_path
    
    # Load data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        data, entity_to_idx, class_to_idx, relation_to_idx = load_data(args.data_path)
        
        # Split into train/val/test
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        test_size = len(data) - train_size - val_size
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        print(f"Data loaded: {len(data)} samples")
        print(f"  - Train: {len(train_data)} samples")
        print(f"  - Val: {len(val_data)} samples")
        print(f"  - Test: {len(test_data)} samples")
        print(f"  - {len(entity_to_idx)} entities")
        print(f"  - {len(class_to_idx)} classes")
        print(f"  - {len(relation_to_idx)} relations")
        
        # Initialize model
        model = RRN(
            embedding_dim=args.embedding_dim,
            num_classes=len(class_to_idx),
            num_relations=len(relation_to_idx)
        )
        
        # Train model
        print("Training model...")
        train_rrn(
            model, 
            train_data, 
            val_data=val_data,
            num_epochs=args.num_epochs, 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            log_dir=args.log_dir
        )
        
        # Evaluate model
        val_acc = evaluate_rrn(model, val_data)
        test_acc = evaluate_rrn(model, test_data)
        
        print(f"Final validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
    else:
        print("No data path provided. Use --data_path or --generate_data")

if __name__ == "__main__":
    main()
