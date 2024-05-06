class Subject:
    def __init__(self, subjectId, disease, icd10):
        self.subjectId = subjectId
        self.disease = disease
        self.hasIcd10 = icd10 != None
        self.icd10 = None if not self.hasIcd10 else icd10.replace("ICD10CM:", "")
        self.isControl = disease == "control"
        self.isSick = not self.isControl
        self.icdFirstLetter = self.icd10[0] if self.hasIcd10 else ("CTL" if self.isControl else "NC")
        self.phenotypes = []
        self.subjectMetrics = {}

    def __repr__(self):
        return f"Subject(subjectId={self.subjectId}, disease={self.disease}, icd10={self.icd10})"
    
class ValidationSubject:
    def __init__(self, subjectId):
        self.subjectId = subjectId
        self.phenotypes = []

    def __repr__(self):
        return f"ValidationSubject(subjectId={self.subjectId})"

class Phenotypes:
    def __init__(self, subjectId, phenotypes):
        self.subjectId = subjectId
        self.phenotypes = phenotypes

    def __repr__(self):
        return f"Phenotypes(subjectId={self.subjectId}, phenotypes={self.phenotypes})"
    
def get_phenotypes(session, subjectId):
    query = """MATCH (a:Biological_sample {subjectid:\"""" + subjectId + """\"})-[:HAS_PHENOTYPE]->(p:Phenotype) 
RETURN a.subjectid as subjectId, collect(p.id) as phenotypes"""
    data = session.run(query).data()
    if len(data) == 0:
        return Phenotypes(subjectId=subjectId, phenotypes=[])
    return Phenotypes(**data[0])

class SubjectMetrics:
    def __init__(self, subjectId, subjectMetrics):
        self.subjectId = subjectId
        self.subjectMetrics = subjectMetrics

    def __repr__(self):
        return f"SubjectMetrics(subjectId={self.subjectId}, numProteins={self.numProteins}, avgProteinScore={self.avgProteinScore}, minProteinScore={self.minProteinScore}, maxProteinScore={self.maxProteinScore}, sumProteinScore={self.sumProteinScore}, numGenes={self.numGenes}, avgGeneScore={self.avgGeneScore}, minGeneScore={self.minGeneScore}, maxGeneScore={self.maxGeneScore}, sumGeneScore={self.sumGeneScore}, numPhenotypes={self.numPhenotypes})"
    
def get_subject_metrics(session, subjectId):
    query = """MATCH (bs:Biological_sample {subjectid:\"""" + subjectId + """\"})
    OPTIONAL MATCH (bs {subjectid:\"""" + subjectId + """\"})-[r_protein:HAS_PROTEIN]->()
    OPTIONAL MATCH (bs {subjectid:\"""" + subjectId + """\"})-[r_damage:HAS_DAMAGE]->()
    OPTIONAL MATCH (bs {subjectid:\"""" + subjectId + """\"})-[r_phenotype:HAS_PHENOTYPE]->()
    WITH bs,
        // Aggregations for Proteins
        COUNT(DISTINCT CASE WHEN r_protein IS NOT NULL THEN r_protein END) AS numProteins,
        AVG(CASE WHEN r_protein IS NOT NULL THEN r_protein.score END) AS avgProteinScore,
        MIN(CASE WHEN r_protein IS NOT NULL THEN r_protein.score END) AS minProteinScore,
        MAX(CASE WHEN r_protein IS NOT NULL THEN r_protein.score END) AS maxProteinScore,
        SUM(CASE WHEN r_protein IS NOT NULL THEN r_protein.score END) AS sumProteinScore,

        // Aggregations for Genes
        COUNT(DISTINCT CASE WHEN r_damage IS NOT NULL THEN r_damage END) AS numGenes,
        AVG(CASE WHEN r_damage IS NOT NULL THEN r_damage.score END) AS avgGeneScore,
        MIN(CASE WHEN r_damage IS NOT NULL THEN r_damage.score END) AS minGeneScore,
        MAX(CASE WHEN r_damage IS NOT NULL THEN r_damage.score END) AS maxGeneScore,
        SUM(CASE WHEN r_damage IS NOT NULL THEN r_damage.score END) AS sumGeneScore,

        // Count of Phenotypes
        COUNT(DISTINCT r_phenotype) AS numPhenotypes

    RETURN bs.subjectid AS subjectId,
       {numProteins: numProteins,
        avgProteinScore: avgProteinScore,
        minProteinScore: minProteinScore,
        maxProteinScore: maxProteinScore,
        sumProteinScore: sumProteinScore,
        numGenes: numGenes,
        avgGeneScore: avgGeneScore,
        minGeneScore: minGeneScore,
        maxGeneScore: maxGeneScore,
        sumGeneScore: sumGeneScore,
        numPhenotypes: numPhenotypes} as subjectMetrics"""
    data = session.run(query).data()
    if len(data) == 0:
        return SubjectMetrics(subjectMetrics={})
    return SubjectMetrics(**data[0])

class DataFetcher:
    def __init__(self, session):
        self.session = session
        self.subjects = self.fetch()
    
    def fetch(self):
        subjects = self.get_subjects(self.session)
        for subject in subjects:
            subject.phenotypes = get_phenotypes(self.session, subject.subjectId).phenotypes
            subject.subjectMetrics = get_subject_metrics(self.session, subject.subjectId).subjectMetrics
        return subjects

    def get_subjects(self, session):
        query = """MATCH (b:Biological_sample)-->(d:Disease)
    WITH *, [s in d.synonyms WHERE s STARTS WITH "ICD10CM" | s] as ICD10
    RETURN b.subjectid as subjectId, d.name as disease, ICD10[0] as icd10"""     
        data = session.run(query).data()
        subjects = [Subject(**record) for record in data]
        return subjects

class ValidationDataFetcher:
    def __init__(self, session):
        self.session = session
        self.subjects = self.fetch()
    
    def fetch(self):
        subjects = self.get_subjects(self.session)
        for subject in subjects:
            subject.phenotypes = get_phenotypes(self.session, subject.subjectId).phenotypes
            subject.subjectMetrics = get_subject_metrics(self.session, subject.subjectId).subjectMetrics

        return subjects

    def get_subjects(self, session):
        query = """MATCH (b:Biological_sample)
    WHERE NOT (b)-[:HAS_DISEASE]-(:Disease)
    RETURN b.subjectid as subjectId"""     
        data = session.run(query).data()
        subjects = [ValidationSubject(**record) for record in data]
        return subjects