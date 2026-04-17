import { useState } from 'react'

import './App.css'

interface Source {
  url: string
  title: string
  heading: string
}

interface SearchResponse {
  answer: string
  sources: Source[]
  chunks: any[]
}

function App() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, k: 5 }),
      })
      
      if (!res.ok) {
        throw new Error('Search failed')
      }
      
      const data = await res.json()
      setResponse(data)
    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Neural Search Engine</h1>
      </header>
      
      <main>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
            className="search-input"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {response && (
          <div className="results">
            <div className="answer">
              <h2>Answer</h2>
              <div className="answer-content">{response.answer}</div>
            </div>

            {response.sources.length > 0 && (
              <div className="sources">
                <h3>Sources</h3>
                <ul>
                  {response.sources.map((source, index) => (
                    <li key={index}>
                      <a href={source.url} target="_blank" rel="noopener noreferrer">
                        {source.title} - {source.heading}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
