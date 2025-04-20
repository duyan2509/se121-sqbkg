import json
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def create_law(tx, law_data):
    tx.run("""
        CREATE (l:Law {id: $id, title: $title, date: $date, source: $source})
    """, id=law_data['id'], title=law_data['title'], date=law_data['date'],
           source=law_data['source'])


def create_chapter(tx, chapter_data, law_id):
    tx.run("""
        CREATE (c:Chapter {id: $id, name: $name, number: $number})
    """, id=chapter_data['id'], name=chapter_data['name'], number=chapter_data['number'])
    tx.run("""
        MATCH (l:Law {id: $law_id}), (c:Chapter {id: $chapter_id})
        CREATE (l)-[:HAS_CHAPTER]->(c)
    """, law_id=law_id, chapter_id=chapter_data['id'])


def create_section(tx, section_name, chapter_id):
    section_id = f"{chapter_id}_section_{section_name.replace(' ', '_').lower()}"
    tx.run("""
        MERGE (s:Section {id: $id})
        ON CREATE SET s.name = $name
    """, id=section_id, name=section_name)
    tx.run("""
        MATCH (c:Chapter {id: $chapter_id}), (s:Section {id: $section_id})
        MERGE (c)-[:HAS_SECTION]->(s)
    """, chapter_id=chapter_id, section_id=section_id)
    return section_id


def create_article(tx, article_data, chapter_id):
    tx.run("""
        CREATE (a:Article {id: $id, parent: $parent, name: $name, number: $number, content: $content})
    """, id=article_data['id'], parent=article_data['parent'], name=article_data['name'],
           number=article_data['number'], content=article_data['content'])

    if article_data['parent'] is not None:
        section_id = create_section(tx, article_data['parent'], chapter_id)
        tx.run("""
            MATCH (s:Section {id: $section_id}), (a:Article {id: $article_id})
            CREATE (s)-[:HAS_ARTICLE]->(a)
        """, section_id=section_id, article_id=article_data['id'])
    else:
        tx.run("""
            MATCH (c:Chapter {id: $chapter_id}), (a:Article {id: $article_id})
            CREATE (c)-[:HAS_ARTICLE]->(a)
        """, chapter_id=chapter_id, article_id=article_data['id'])


def create_clause(tx, clause_data, article_id):
    tx.run("""
        CREATE (cl:Clause {id: $id, number: $number, content: $content})
    """, id=clause_data['id'], number=clause_data['number'], content=clause_data['content'])
    tx.run("""
        MATCH (a:Article {id: $article_id}), (cl:Clause {id: $clause_id})
        CREATE (a)-[:HAS_CLAUSE]->(cl)
    """, article_id=article_id, clause_id=clause_data['id'])


def create_point(tx, point_data, clause_id):
    tx.run("""
        CREATE (p:Point {id: $id, number: $number, content: $content})
    """, id=point_data['id'], number=point_data['number'], content=point_data['content'])
    tx.run("""
        MATCH (cl:Clause {id: $clause_id}), (p:Point {id: $point_id})
        CREATE (cl)-[:HAS_POINT]->(p)
    """, clause_id=clause_id, point_id=point_data['id'])


def create_reference(tx, article_id, reference_id):
    tx.run("""
        MATCH (a:Article {id: $article_id}), (ref:Article {id: $reference_id})
        CREATE (a)-[:REFERENCES]->(ref)
    """, article_id=article_id, reference_id=reference_id)


def import_law_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    law_data = data['law']

    with driver.session() as session:
        session.execute_write(create_law, law_data)

        for chapter in law_data['chapters']:
            session.execute_write(create_chapter, chapter, law_data['id'])

            for article in chapter['articles']:
                session.execute_write(create_article, article, chapter['id'])

                for clause in article['clauses']:
                    session.execute_write(create_clause, clause, article['id'])

                    for point in clause.get('points', []):
                        session.execute_write(create_point, point, clause['id'])

                for reference in article.get('references', []):
                    session.execute_write(create_reference, article['id'], reference)


file_path = 'law_structure_from_txt.json'
import_law_data(file_path)

driver.close()