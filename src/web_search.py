from typing import List
import aiohttp
from bs4 import BeautifulSoup

from .models import SearchResult, SourceType
from .config import AppSettings
from .logging_config import get_logger

logger = get_logger(__name__)

class WebSearchProvider:    
    def __init__(self, settings: AppSettings):
        self.timeout = 10
        self.session = None
        logger.info("[SEARCH] Using free web search (DuckDuckGo + Bing)")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        results = []
        
        try:
            ddg_results = await self._search_duckduckgo(query, max_results)
            results.extend(ddg_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        if len(results) < max_results:
            try:
                bing_results = await self._search_bing(query, max_results - len(results))
                results.extend(bing_results)
            except Exception as e:
                logger.warning(f"Bing search failed: {e}")
        
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.source not in seen_urls and len(unique_results) < max_results:
                seen_urls.add(result.source)
                unique_results.append(result)
        
        return unique_results
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with self.session.get(search_url, headers=headers, timeout=self.timeout) as response:
            if response.status != 200:
                return []
            
            html_content = await response.text()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        result_divs = soup.find_all('div', class_='result')[:max_results]
        
        for i, div in enumerate(result_divs):
            try:
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('div', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append(SearchResult(
                        content=f"{title_elem.get_text().strip()}\n{snippet_elem.get_text().strip()}",
                        source=title_elem.get('href', ''),
                        score=1.0 - (i * 0.1),
                        metadata={
                            'title': title_elem.get_text().strip(),
                            'snippet': snippet_elem.get_text().strip(),
                            'search_engine': 'duckduckgo',
                            'search_query': query
                        },
                        source_type=SourceType.WEB,
                        rank=i + 1
                    ))
            except Exception:
                continue
        
        return results
    
    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            async with self.session.get(search_url, headers=headers, timeout=self.timeout) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            result_items = soup.find_all('li', class_='b_algo')[:max_results]
            
            for i, item in enumerate(result_items):
                try:
                    title_elem = item.find('h2')
                    link_elem = title_elem.find('a') if title_elem else None
                    snippet_elem = item.find('div', class_='b_caption')
                    
                    if link_elem and snippet_elem:
                        title = title_elem.get_text().strip()
                        url = link_elem.get('href', '')
                        snippet = snippet_elem.get_text().strip()
                        
                        results.append(SearchResult(
                            content=f"{title}\n{snippet}",
                            source=url,
                            score=1.0 - (i * 0.1),
                            metadata={
                                'title': title,
                                'snippet': snippet,
                                'search_engine': 'bing',
                                'search_query': query
                            },
                            source_type=SourceType.WEB,
                            rank=i + 1 + 10
                        ))
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Bing search error: {e}")
            return []