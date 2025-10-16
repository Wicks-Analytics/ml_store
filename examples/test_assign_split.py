"""Simple test script to verify assign_split functionality."""

import polars as pl
from ml_store import assign_split


def test_random_split():
    """Test random splitting."""
    print("=" * 60)
    print("Testing Random Split")
    print("=" * 60)
    
    # Create sample data
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)]
    })
    
    print(f"Original dataset: {df.shape[0]} rows\n")
    
    # Config without split column (will use random split)
    config = {
        "model_type": "classifier",
        "target_column": "target"
    }
    
    # Split the data
    train_df, test_df, holdout_df = assign_split(
        df,
        config,
        train_ratio=0.7,
        test_ratio=0.15,
        holdout_ratio=0.15,
        random_seed=42
    )
    
    print(f"Train set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Holdout set: {holdout_df.shape[0]} rows ({holdout_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Total: {train_df.shape[0] + test_df.shape[0] + holdout_df.shape[0]} rows")
    print("\n✓ Random split test passed!\n")


def test_column_split():
    """Test column-based splitting."""
    print("=" * 60)
    print("Testing Column-Based Split")
    print("=" * 60)
    
    # Create sample data with split column
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)],
        "split": ["train"] * 70 + ["test"] * 15 + ["holdout"] * 15
    })
    
    print(f"Original dataset: {df.shape[0]} rows")
    print(f"Split column values: {df['split'].value_counts()}\n")
    
    # Config with split column
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "split_column": "split",
        "split_values": {
            "train": "train",
            "test": "test",
            "holdout": "holdout"
        }
    }
    
    # Split the data
    train_df, test_df, holdout_df = assign_split(df, config)
    
    print(f"Train set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Holdout set: {holdout_df.shape[0]} rows ({holdout_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Total: {train_df.shape[0] + test_df.shape[0] + holdout_df.shape[0]} rows")
    
    # Verify splits match the column values
    assert train_df.shape[0] == 70, "Train set size mismatch"
    assert test_df.shape[0] == 15, "Test set size mismatch"
    assert holdout_df.shape[0] == 15, "Holdout set size mismatch"
    
    print("\n✓ Column-based split test passed!\n")


def test_custom_split_values():
    """Test column-based splitting with custom values."""
    print("=" * 60)
    print("Testing Custom Split Values")
    print("=" * 60)
    
    # Create sample data with custom split values
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)],
        "dataset": ["TRAIN"] * 70 + ["TEST"] * 15 + ["HOLDOUT"] * 15
    })
    
    print(f"Original dataset: {df.shape[0]} rows")
    print(f"Split column values: {df['dataset'].value_counts()}\n")
    
    # Config with custom split values
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "split_column": "dataset",
        "split_values": {
            "train": "TRAIN",
            "test": "TEST",
            "holdout": "HOLDOUT"
        }
    }
    
    # Split the data
    train_df, test_df, holdout_df = assign_split(df, config)
    
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    print(f"Holdout set: {holdout_df.shape[0]} rows")
    
    # Verify splits
    assert train_df.shape[0] == 70, "Train set size mismatch"
    assert test_df.shape[0] == 15, "Test set size mismatch"
    assert holdout_df.shape[0] == 15, "Holdout set size mismatch"
    
    print("\n✓ Custom split values test passed!\n")


def test_config_ratios():
    """Test ratios from config file."""
    print("=" * 60)
    print("Testing Config-Based Ratios")
    print("=" * 60)
    
    # Create sample data
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)]
    })
    
    print(f"Original dataset: {df.shape[0]} rows\n")
    
    # Config with ratios specified
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "train_ratio": 0.6,
        "test_ratio": 0.2,
        "holdout_ratio": 0.2
    }
    
    print(f"Config ratios: train={config['train_ratio']}, test={config['test_ratio']}, holdout={config['holdout_ratio']}\n")
    
    # Split the data - config ratios should override function parameters
    train_df, test_df, holdout_df = assign_split(
        df,
        config,
        train_ratio=0.7,  # This should be ignored
        test_ratio=0.15,  # This should be ignored
        holdout_ratio=0.15,  # This should be ignored
        random_seed=42
    )
    
    print(f"Train set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Holdout set: {holdout_df.shape[0]} rows ({holdout_df.shape[0]/df.shape[0]*100:.1f}%)")
    
    # Verify that config ratios were used (60/20/20 split)
    assert train_df.shape[0] == 60, f"Expected 60 train rows, got {train_df.shape[0]}"
    assert test_df.shape[0] == 20, f"Expected 20 test rows, got {test_df.shape[0]}"
    assert holdout_df.shape[0] == 20, f"Expected 20 holdout rows, got {holdout_df.shape[0]}"
    
    print("\n✓ Config-based ratios test passed!\n")


def test_config_random_seed():
    """Test random_seed from config file."""
    print("=" * 60)
    print("Testing Config-Based Random Seed")
    print("=" * 60)
    
    # Create sample data
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)]
    })
    
    print(f"Original dataset: {df.shape[0]} rows\n")
    
    # Config with random_seed specified
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "random_seed": 999
    }
    
    print(f"Config random_seed: {config['random_seed']}\n")
    
    # Split the data twice with config seed - should get same results
    train_df1, test_df1, holdout_df1 = assign_split(df, config)
    train_df2, test_df2, holdout_df2 = assign_split(df, config)
    
    # Verify reproducibility
    assert train_df1.equals(train_df2), "Train sets should be identical with same seed"
    assert test_df1.equals(test_df2), "Test sets should be identical with same seed"
    assert holdout_df1.equals(holdout_df2), "Holdout sets should be identical with same seed"
    
    print(f"Train set: {train_df1.shape[0]} rows")
    print(f"Test set: {test_df1.shape[0]} rows")
    print(f"Holdout set: {holdout_df1.shape[0]} rows")
    print("Reproducibility verified with config seed!")
    
    print("\n✓ Config-based random_seed test passed!\n")


def test_default_ratios():
    """Test default ratios when nothing is specified."""
    print("=" * 60)
    print("Testing Default Ratios")
    print("=" * 60)
    
    # Create sample data
    df = pl.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": [i % 2 for i in range(100)]
    })
    
    print(f"Original dataset: {df.shape[0]} rows\n")
    
    # Config without ratios or split column
    config = {
        "model_type": "classifier",
        "target_column": "target"
    }
    
    # Split the data - should use defaults (0.7, 0.15, 0.15)
    train_df, test_df, holdout_df = assign_split(df, config, random_seed=42)
    
    print(f"Train set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Holdout set: {holdout_df.shape[0]} rows ({holdout_df.shape[0]/df.shape[0]*100:.1f}%)")
    
    # Verify defaults were used (70/15/15 split)
    assert train_df.shape[0] == 70, f"Expected 70 train rows, got {train_df.shape[0]}"
    assert test_df.shape[0] == 15, f"Expected 15 test rows, got {test_df.shape[0]}"
    assert holdout_df.shape[0] == 15, f"Expected 15 holdout rows, got {holdout_df.shape[0]}"
    
    print("\n✓ Default ratios test passed!\n")


if __name__ == "__main__":
    test_random_split()
    test_column_split()
    test_custom_split_values()
    test_config_ratios()
    test_config_random_seed()
    test_default_ratios()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
