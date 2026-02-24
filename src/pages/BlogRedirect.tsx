import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";

// Maps old URL patterns to new blog IDs
const urlToBlogIdMap: Record<string, string> = {
  "/llms/distributed%20ml/optimization/2025/10/02/efficient-ml.html": "llm-optimization",
  "/llms/distributed ml/optimization/2025/10/02/efficient-ml.html": "llm-optimization",
};

export default function BlogRedirect() {
  const navigate = useNavigate();
  const location = useLocation();
  
  useEffect(() => {
    // Get the full pathname
    const pathname = location.pathname;
    
    // Check if we have a mapping for this path
    const blogId = urlToBlogIdMap[pathname];
    
    if (blogId) {
      // Redirect to the new URL
      navigate(`/blogs/${blogId}`, { replace: true });
    } else {
      // If no mapping found, redirect to blogs list
      navigate("/blogs", { replace: true });
    }
  }, [navigate, location]);

  return null; // This component just redirects, no UI needed
}
