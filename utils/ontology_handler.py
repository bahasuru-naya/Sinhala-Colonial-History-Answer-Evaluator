"""
Ontology Handler for Colonial Sri Lanka
Manages RDF graph loading and querying
"""
import logging
from typing import List, Dict, Optional, Set
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS
from rdflib.namespace import OWL, XSD
from pathlib import Path
from config import ONTOLOGY_FILE, ONTOLOGY_NS

logger = logging.getLogger(__name__)


class OntologyHandler:
    """Handles RDF ontology for Colonial Sri Lanka history"""
    
    def __init__(self, ontology_file: Path = ONTOLOGY_FILE):
        """
        Initialize ontology handler
        
        Args:
            ontology_file: Path to RDF/TTL ontology file
        """
        self.ontology_file = Path(ontology_file)
        self.graph = Graph()
        self.ns = Namespace(ONTOLOGY_NS)
        
        self.load_ontology()
    
    def load_ontology(self) -> bool:
        """
        Load RDF ontology from file
        
        Returns:
            Success status
        """
        try:
            if not self.ontology_file.exists():
                logger.warning(f"Ontology file not found: {self.ontology_file}")
                return False
            
            self.graph.parse(str(self.ontology_file), format='turtle')
            logger.info(f"✓ Loaded ontology: {len(self.graph)} triples")
            return True
        
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            return False
    
    def get_entity_description(self, entity_uri: str) -> Optional[str]:
        """
        Get description of an entity from ontology
        
        Args:
            entity_uri: URI or name of entity (e.g., 'PortuguesePeriod')
            
        Returns:
            Description text or None
        """
        try:
            # Try different URI formats
            if not entity_uri.startswith('http'):
                entity = self.ns[entity_uri]
            else:
                entity = URIRef(entity_uri)
            
            # Query for description/comment
            descriptions = list(self.graph.objects(entity, RDFS.comment))
            if descriptions:
                return str(descriptions[0])
            
            # Fallback to label
            labels = list(self.graph.objects(entity, RDFS.label))
            if labels:
                return str(labels[0])
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting entity description: {e}")
            return None
    
    def get_related_entities(self, entity_uri: str, relation_type: Optional[str] = None) -> List[Dict]:
        """
        Get entities related to a given entity
        
        Args:
            entity_uri: URI or name of entity
            relation_type: Optional specific relation type to filter
            
        Returns:
            List of related entities with details
        """
        try:
            if not entity_uri.startswith('http'):
                entity = self.ns[entity_uri]
            else:
                entity = URIRef(entity_uri)
            
            related = []
            
            # Get outgoing relations
            for predicate, obj in self.graph.predicate_objects(entity):
                if relation_type and predicate != self.ns[relation_type]:
                    continue
                
                if isinstance(obj, URIRef):
                    label = self._get_label(obj)
                    related.append({
                        'relation': self._get_label(predicate),
                        'target': label,
                        'target_uri': str(obj)
                    })
            
            return related
        
        except Exception as e:
            logger.error(f"Error getting related entities: {e}")
            return []
    
    def get_period_entities(self, period: str) -> List[Dict]:
        """
        Get all entities (events, rulers, systems) for a specific period
        
        Args:
            period: Period name (e.g., 'PortuguesePeriod', 'BritishPeriod')
            
        Returns:
            List of entities and their descriptions
        """
        try:
            if not period.startswith('http'):
                period_uri = self.ns[period]
            else:
                period_uri = URIRef(period)
            
            entities = []
            
            # Query for entities that occurred during this period
            query = f"""
            SELECT DISTINCT ?entity ?label ?description
            WHERE {{
                ?entity <{self.ns}occurredDuring> <{period_uri}> .
                OPTIONAL {{ ?entity <{RDFS.label}> ?label }}
                OPTIONAL {{ ?entity <{RDFS.comment}> ?description }}
            }}
            """
            
            results = self.graph.query(query)
            for row in results:
                entities.append({
                    'entity': str(row.entity),
                    'label': str(row.label) if row.label else str(row.entity).split('/')[-1],
                    'description': str(row.description) if row.description else '',
                    'period': period
                })
            
            return entities
        
        except Exception as e:
            logger.error(f"Error getting period entities: {e}")
            return []
    
    def get_commodities(self, economic_system: Optional[str] = None) -> List[Dict]:
        """
        Get commodities traded during colonial period
        
        Args:
            economic_system: Optional specific economic system to filter
            
        Returns:
            List of commodities with details
        """
        try:
            commodities = []
            
            if economic_system:
                if not economic_system.startswith('http'):
                    system_uri = self.ns[economic_system]
                else:
                    system_uri = URIRef(economic_system)
                
                # Get commodities for specific system
                query = f"""
                SELECT DISTINCT ?commodity ?label
                WHERE {{
                    <{system_uri}> <{self.ns}tradedCommodity> ?commodity .
                    OPTIONAL {{ ?commodity <{RDFS.label}> ?label }}
                }}
                """
            else:
                # Get all commodities
                query = f"""
                SELECT DISTINCT ?commodity ?label
                WHERE {{
                    ?commodity <{RDF.type}> <{self.ns}Commodity> .
                    OPTIONAL {{ ?commodity <{RDFS.label}> ?label }}
                }}
                """
            
            results = self.graph.query(query)
            for row in results:
                commodities.append({
                    'uri': str(row.commodity),
                    'name': str(row.label) if row.label else str(row.commodity).split('/')[-1]
                })
            
            return commodities
        
        except Exception as e:
            logger.error(f"Error getting commodities: {e}")
            return []
    
    def search_entities(self, query_term: str) -> List[Dict]:
        """
        Search for entities matching a query term
        
        Args:
            query_term: Term to search for
            
        Returns:
            List of matching entities
        """
        try:
            results = []
            query_lower = query_term.lower()
            
            # Search in labels and descriptions
            for s, p, o in self.graph:
                if isinstance(o, Literal):
                    if query_lower in str(o).lower():
                        subject_label = self._get_label(s)
                        results.append({
                            'entity': subject_label,
                            'entity_uri': str(s),
                            'match_text': str(o)[:100]
                        })
            
            # Remove duplicates
            unique_results = []
            seen = set()
            for r in results:
                if r['entity'] not in seen:
                    unique_results.append(r)
                    seen.add(r['entity'])
            
            return unique_results
        
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
    
    def get_historical_timeline(self) -> List[Dict]:
        """
        Get chronological timeline of periods and events
        
        Returns:
            Sorted list of periods and events with dates
        """
        try:
            timeline = []
            
            # Query for all entities with start years
            query = f"""
            SELECT DISTINCT ?entity ?label ?startYear ?endYear ?type
            WHERE {{
                ?entity <{self.ns}startYear> ?startYear .
                OPTIONAL {{ ?entity <{RDFS.label}> ?label }}
                OPTIONAL {{ ?entity <{self.ns}endYear> ?endYear }}
                OPTIONAL {{ ?entity <{RDF.type}> ?type }}
            }}
            ORDER BY ?startYear
            """
            
            results = self.graph.query(query)
            for row in results:
                timeline.append({
                    'entity': str(row.label) if row.label else str(row.entity).split('/')[-1],
                    'entity_uri': str(row.entity),
                    'start_year': int(row.startYear) if row.startYear else None,
                    'end_year': int(row.endYear) if row.endYear else None,
                    'type': str(row.type).split('/')[-1] if row.type else None
                })
            
            return timeline
        
        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return []
    
    def _get_label(self, uri: URIRef) -> str:
        """Get human-readable label for a URI"""
        try:
            labels = list(self.graph.objects(uri, RDFS.label))
            if labels:
                return str(labels[0])
            return str(uri).split('/')[-1]
        except:
            return str(uri).split('/')[-1]
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract key historical concepts from text using ontology
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relevant ontology concepts found in text
        """
        concepts = []
        text_lower = text.lower()
        
        try:
            # Search for entity labels in text
            for s, p, o in self.graph:
                if isinstance(o, Literal) and isinstance(s, URIRef):
                    label = str(o).lower()
                    if len(label) > 3 and label in text_lower:
                        entity_name = self._get_label(s)
                        if entity_name not in concepts:
                            concepts.append(entity_name)
            
            return concepts
        
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []


def initialize_ontology() -> OntologyHandler:
    """Initialize and return ontology handler"""
    return OntologyHandler()
