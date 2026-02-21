"""
Database initialization script.
"""
import asyncio
import asyncpg
from core.config import settings

async def create_tables():
    """Initialize all database tables"""
    
    conn = await asyncpg.connect(settings.DATABASE_URL)
    
    try:
        # Enable UUID extension
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        
        # Users table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(50) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                subscription_tier VARCHAR(20) DEFAULT 'free',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE,
                daily_request_count INTEGER DEFAULT 0,
                monthly_request_count INTEGER DEFAULT 0,
                last_request_reset DATE DEFAULT CURRENT_DATE,
                preferred_framework VARCHAR(20) DEFAULT 'qiskit',
                preferred_language VARCHAR(10) DEFAULT 'en',
                theme VARCHAR(10) DEFAULT 'dark'
            );
        ''')
        
        # API Requests table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS api_requests (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                endpoint VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50) NOT NULL,
                request_data JSONB,
                framework_source VARCHAR(20),
                framework_target VARCHAR(20),
                response_data JSONB,
                status_code INTEGER,
                error_message TEXT,
                latency_ms INTEGER,
                llm_tokens_used INTEGER,
                rag_documents_retrieved INTEGER,
                cache_hit BOOLEAN DEFAULT FALSE,
                validation_passed BOOLEAN,
                confidence_score DECIMAL(3, 2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at TIMESTAMP WITH TIME ZONE,
                cost_credits DECIMAL(10, 4) DEFAULT 0.0
            );
        ''')
        
        # Code Snippets table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(200),
                description TEXT,
                framework VARCHAR(20) NOT NULL,
                code_content TEXT NOT NULL,
                input_prompt TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                is_favorite BOOLEAN DEFAULT FALSE,
                tags TEXT[],
                share_token VARCHAR(100) UNIQUE,
                view_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        ''')
        
        # Subscriptions table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                tier VARCHAR(20) NOT NULL,
                stripe_customer_id VARCHAR(100),
                stripe_subscription_id VARCHAR(100),
                monthly_request_limit INTEGER,
                daily_request_limit INTEGER,
                status VARCHAR(20) DEFAULT 'active',
                trial_ends_at TIMESTAMP WITH TIME ZONE,
                current_period_start TIMESTAMP WITH TIME ZONE,
                current_period_end TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        ''')
        
        # Feedback table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                request_id UUID REFERENCES api_requests(id) ON DELETE CASCADE,
                rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                feedback_type VARCHAR(50),
                comment TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);',
            'CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_tier);',
            'CREATE INDEX IF NOT EXISTS idx_requests_user_id ON api_requests(user_id);',
            'CREATE INDEX IF NOT EXISTS idx_requests_feature ON api_requests(feature_type);',
            'CREATE INDEX IF NOT EXISTS idx_requests_created_at ON api_requests(created_at);',
            'CREATE INDEX IF NOT EXISTS idx_snippets_user_id ON code_snippets(user_id);',
            'CREATE INDEX IF NOT EXISTS idx_snippets_public ON code_snippets(is_public);',
            'CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);',
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        print("✅ Database tables created successfully!")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_tables())
