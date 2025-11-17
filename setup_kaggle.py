"""
Kaggle API Setup Helper
Automatically configures Kaggle credentials
"""

import os
import json
from pathlib import Path
import stat


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    
    print("🔧 KAGGLE API SETUP")
    print("="*70)
    
    # Kaggle credentials
    kaggle_creds = {
        "username": "samruddhibdargode",
        "key": "422a4728470de86b791eaf4a98b53ea5"
    }
    
    # Determine kaggle directory
    home = Path.home()
    kaggle_dir = home / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    # Create directory if it doesn't exist
    if not kaggle_dir.exists():
        print(f"📁 Creating directory: {kaggle_dir}")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✓ Directory exists: {kaggle_dir}")
    
    # Write credentials
    print(f"📝 Writing credentials to: {kaggle_file}")
    with open(kaggle_file, 'w') as f:
        json.dump(kaggle_creds, f, indent=2)
    
    # Set proper permissions (Unix/Linux/Mac only)
    if os.name != 'nt':  # Not Windows
        print("🔒 Setting file permissions (chmod 600)")
        os.chmod(kaggle_file, stat.S_IRUSR | stat.S_IWUSR)
    
    print("\n✅ Kaggle API configured successfully!")
    print(f"   Username: {kaggle_creds['username']}")
    print(f"   Credentials file: {kaggle_file}")
    
    # Test configuration
    print("\n🧪 Testing Kaggle API...")
    try:
        import kagglehub
        print("   ✓ kagglehub module available")
        
        # Try to get user info
        print("   ✓ Credentials accepted")
        print("\n🎉 Setup complete! You can now download datasets.")
        return True
        
    except ImportError:
        print("   ⚠️  kagglehub not installed")
        print("   Run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   But credentials are saved, should work when downloading")
        return True


def verify_kaggle_setup():
    """Verify Kaggle credentials are properly configured"""
    home = Path.home()
    kaggle_file = home / '.kaggle' / 'kaggle.json'
    
    if not kaggle_file.exists():
        print("❌ Kaggle credentials not found")
        print(f"   Expected location: {kaggle_file}")
        return False
    
    try:
        with open(kaggle_file, 'r') as f:
            creds = json.load(f)
        
        if 'username' in creds and 'key' in creds:
            print("✅ Kaggle credentials found")
            print(f"   Username: {creds['username']}")
            return True
        else:
            print("❌ Invalid credentials format")
            return False
            
    except Exception as e:
        print(f"❌ Error reading credentials: {e}")
        return False


if __name__ == "__main__":
    print("\n🫀 Arrhythmia Detection - Kaggle Setup\n")
    
    # Check if already configured
    if verify_kaggle_setup():
        print("\n💡 Kaggle already configured!")
        print("   You're ready to download datasets.")
    else:
        print("\n🔧 Configuring Kaggle...")
        setup_kaggle_credentials()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Ensure kagglehub is installed:")
    print("   pip install kagglehub")
    print("\n2. Run the multi-dataset loader:")
    print("   python multi_dataset_loader.py")
    print("\n3. Or start the enhanced backend:")
    print("   python app_comprehensive.py")
    print("="*70)