import sqlite3
import pandas as pd
import pickle
import io
import os

def create_tables():
    """
    Create database tables if they don't exist.
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        data BLOB,
        predictions BLOB,
        model BLOB,
        preprocessor BLOB,
        feature_names BLOB,
        model_type TEXT,
        is_training_session BOOLEAN DEFAULT 1,
        used_model_id INTEGER,
        notes TEXT
    )
    ''')
    
    # Create pretrained models table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trained_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_type TEXT NOT NULL,
        model BLOB,
        preprocessor BLOB,
        feature_names BLOB,
        metrics BLOB,
        training_data_size INTEGER,
        notes TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def save_session(name, data, predictions, model, preprocessor, feature_names, model_type=None, is_training_session=True, used_model_id=None, notes=None):
    """
    Save a session to the database.
    
    Args:
        name: Session name
        data: DataFrame with original data
        predictions: DataFrame with predictions
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        model_type: Type of model used (XGBoost, RandomForest, etc.)
        is_training_session: Whether this session included model training
        used_model_id: ID of pretrained model used for prediction (if not training)
        notes: Additional notes about the session
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    # Serialize the data
    data_bytes = pickle.dumps(data)
    predictions_bytes = pickle.dumps(predictions)
    model_bytes = pickle.dumps(model) if model is not None else None
    preprocessor_bytes = pickle.dumps(preprocessor) if preprocessor is not None else None
    feature_names_bytes = pickle.dumps(feature_names) if feature_names is not None else None
    
    # Check if session with the same name exists
    cursor.execute('SELECT id FROM sessions WHERE name = ?', (name,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing session
        cursor.execute('''
        UPDATE sessions 
        SET data = ?, predictions = ?, model = ?, preprocessor = ?, feature_names = ?, 
            model_type = ?, is_training_session = ?, used_model_id = ?, notes = ?,
            created_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (data_bytes, predictions_bytes, model_bytes, preprocessor_bytes, feature_names_bytes, 
              model_type, is_training_session, used_model_id, notes, existing[0]))
    else:
        # Insert new session
        cursor.execute('''
        INSERT INTO sessions (name, data, predictions, model, preprocessor, feature_names, 
                            model_type, is_training_session, used_model_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, data_bytes, predictions_bytes, model_bytes, preprocessor_bytes, feature_names_bytes,
              model_type, is_training_session, used_model_id, notes))
    
    conn.commit()
    conn.close()

def load_sessions():
    """
    Load all session names and IDs.
    
    Returns:
        List of tuples (id, name, created_at)
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, created_at FROM sessions ORDER BY created_at DESC')
    sessions = cursor.fetchall()
    
    conn.close()
    
    return sessions

def load_session_data(session_id):
    """
    Load data for a specific session.
    
    Args:
        session_id: Session ID
    
    Returns:
        Tuple of (data, predictions, model, preprocessor, feature_names, model_type, is_training_session, used_model_id, notes)
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT data, predictions, model, preprocessor, feature_names, 
           model_type, is_training_session, used_model_id, notes 
    FROM sessions WHERE id = ?
    ''', (session_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        data = pickle.loads(result[0]) if result[0] is not None else None
        predictions = pickle.loads(result[1]) if result[1] is not None else None
        model = pickle.loads(result[2]) if result[2] is not None else None
        preprocessor = pickle.loads(result[3]) if result[3] is not None else None
        feature_names = pickle.loads(result[4]) if result[4] is not None else None
        model_type = result[5]
        is_training_session = result[6]
        used_model_id = result[7]
        notes = result[8]
        
        return data, predictions, model, preprocessor, feature_names, model_type, is_training_session, used_model_id, notes
    
    return None, None, None, None, None, None, None, None, None

def delete_session(session_id):
    """
    Delete a session from the database.
    
    Args:
        session_id: Session ID
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    
    conn.commit()
    conn.close()

def save_trained_model(name, model_type, model, preprocessor, feature_names, metrics=None, training_data_size=None, notes=None):
    """
    Save a trained model to the database.
    
    Args:
        name: Model name
        model_type: Type of model (XGBoost, RandomForest, etc.)
        model: Trained model object
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        metrics: Dictionary with evaluation metrics
        training_data_size: Size of the training dataset
        notes: Additional notes about the model
    
    Returns:
        ID of the saved model
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    # Serialize the data
    model_bytes = pickle.dumps(model)
    preprocessor_bytes = pickle.dumps(preprocessor)
    feature_names_bytes = pickle.dumps(feature_names)
    metrics_bytes = pickle.dumps(metrics) if metrics is not None else None
    
    # Check if model with the same name and type exists
    cursor.execute('SELECT id FROM trained_models WHERE name = ? AND model_type = ?', (name, model_type))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing model
        cursor.execute('''
        UPDATE trained_models 
        SET model = ?, preprocessor = ?, feature_names = ?, 
            metrics = ?, training_data_size = ?, notes = ?,
            created_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (model_bytes, preprocessor_bytes, feature_names_bytes, 
              metrics_bytes, training_data_size, notes, existing[0]))
        model_id = existing[0]
    else:
        # Insert new model
        cursor.execute('''
        INSERT INTO trained_models (name, model_type, model, preprocessor, feature_names, 
                                 metrics, training_data_size, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, model_type, model_bytes, preprocessor_bytes, feature_names_bytes,
              metrics_bytes, training_data_size, notes))
        model_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return model_id

def load_trained_models():
    """
    Load all trained model names and details.
    
    Returns:
        List of tuples (id, name, model_type, created_at, training_data_size)
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, model_type, created_at, training_data_size 
    FROM trained_models 
    ORDER BY created_at DESC
    ''')
    models = cursor.fetchall()
    
    conn.close()
    
    return models

def load_trained_model(model_id):
    """
    Load a trained model from the database.
    
    Args:
        model_id: Model ID
    
    Returns:
        Tuple of (model, preprocessor, feature_names, metrics, model_type)
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT model, preprocessor, feature_names, metrics, model_type
    FROM trained_models WHERE id = ?
    ''', (model_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        model = pickle.loads(result[0])
        preprocessor = pickle.loads(result[1])
        feature_names = pickle.loads(result[2])
        metrics = pickle.loads(result[3]) if result[3] is not None else None
        model_type = result[4]
        
        return model, preprocessor, feature_names, metrics, model_type
    
    return None, None, None, None, None

def delete_trained_model(model_id):
    """
    Delete a trained model from the database.
    
    Args:
        model_id: Model ID
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM trained_models WHERE id = ?', (model_id,))
    
    conn.commit()
    conn.close()

def get_latest_model_by_type(model_type):
    """
    Get the latest trained model by type.
    
    Args:
        model_type: Type of model (XGBoost, RandomForest, etc.)
    
    Returns:
        Tuple of (model_id, model, preprocessor, feature_names)
    """
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, model, preprocessor, feature_names
    FROM trained_models 
    WHERE model_type = ? 
    ORDER BY created_at DESC LIMIT 1
    ''', (model_type,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        model_id = result[0]
        model = pickle.loads(result[1])
        preprocessor = pickle.loads(result[2])
        feature_names = pickle.loads(result[3])
        
        return model_id, model, preprocessor, feature_names
    
    return None, None, None, None
