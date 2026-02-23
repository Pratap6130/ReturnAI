#!/usr/bin/env python3
"""
Quick setup script for local development environment
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        return False


def setup_backend():
    """Setup backend environment"""
    print("🔧 Setting up backend...")
    backend_dir = Path(__file__).parent / "backend"
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating backend .env file...")
        with open(env_example) as f:
            content = f.read()
        with open(env_file, "w") as f:
            f.write(content)
        print("✅ Created backend/.env - Please update with your values")
    
    print("✅ Backend setup complete")


def setup_frontend():
    """Setup frontend environment"""
    print("🔧 Setting up frontend...")
    frontend_dir = Path(__file__).parent / "frontend-react"
    
    # Create .env file if it doesn't exist
    env_file = frontend_dir / ".env"
    env_example = frontend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating frontend .env file...")
        with open(env_example) as f:
            content = f.read()
        with open(env_file, "w") as f:
            f.write(content)
        print("✅ Created frontend-react/.env - Please update with your values")
    
    print("✅ Frontend setup complete")


def main():
    """Main setup function"""
    print("🚀 Return Risk Prediction - Local Development Setup\n")
    
    setup_backend()
    print()
    setup_frontend()
    
    print("\n" + "="*60)
    print("✨ Setup complete!")
    print("="*60)
    print("\n📚 Next steps:")
    print("1. Update environment variables in .env files")
    print("2. Install backend dependencies: pip install -r backend/requirements.txt")
    print("3. Install frontend dependencies: cd frontend-react && npm install")
    print("4. Start backend: cd backend && uvicorn main:app --reload")
    print("5. Start frontend: cd frontend-react && npm run dev")
    print("\n📖 For deployment instructions, see DEPLOYMENT.md")


if __name__ == "__main__":
    main()
