import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ThemeProvider } from "./context/ThemeContext"
import Header from "./components/Header"
import Home from "./pages/Home"
import Blogs from "./pages/Blogs"
import BlogPost from "./pages/BlogPost"
import NewsPage from "./pages/News"

function App() {
  return (
    <ThemeProvider>
      <div className="flex flex-col min-h-screen bg-background text-foreground transition-colors duration-300">
        <BrowserRouter>
          <Header />
          <main className="flex-1 pt-14 sm:pt-16">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/blogs" element={<Blogs />} />
              <Route path="/blogs/:id" element={<BlogPost />} />
              <Route path="/news" element={<NewsPage />} />
            </Routes>
          </main>
        </BrowserRouter>
      </div>
    </ThemeProvider>
  )
}

export default App
