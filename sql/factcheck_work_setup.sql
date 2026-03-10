CREATE SCHEMA IF NOT EXISTS factcheck_work;

CREATE OR REPLACE VIEW factcheck_work.bill_vote_fact_view AS
SELECT
    pv.id AS partyvote_id,
    pv.vote,
    pv.party_id,
    cp.name_en AS party_name,
    vq.id AS votequestion_id,
    vq.date AS vote_date,
    vq.bill_id,
    b.number AS bill_number,
    COALESCE(NULLIF(b.short_title_en, ''), NULLIF(b.name_en, ''), b.number) AS bill_title,
    b.session_id,
    cs.parliamentnum AS parliament_number,
    cs.sessnum AS session_number
FROM public.bills_partyvote pv
JOIN public.bills_votequestion vq ON vq.id = pv.votequestion_id
JOIN public.bills_bill b ON b.id = vq.bill_id
LEFT JOIN public.core_session cs ON cs.id = b.session_id
JOIN public.core_party cp ON cp.id = pv.party_id
WHERE pv.vote IN ('Y', 'N')
  AND vq.bill_id IS NOT NULL
  AND vq.date IS NOT NULL
  AND b.number IS NOT NULL;

CREATE OR REPLACE VIEW factcheck_work.hansard_claim_candidates AS
SELECT
    s.id AS statement_id,
    s.document_id,
    d.date AS source_date,
    s.who_en,
    regexp_replace(s.content_en, '<[^>]+>', ' ', 'g') AS content_text,
    s.bill_debated_id AS bill_id,
    b.number AS bill_number,
    COALESCE(NULLIF(b.short_title_en, ''), NULLIF(b.name_en, ''), b.number) AS bill_title,
    b.session_id,
    cs.parliamentnum AS parliament_number,
    cs.sessnum AS session_number
FROM public.hansards_statement s
JOIN public.hansards_document d ON d.id = s.document_id
JOIN public.bills_bill b ON b.id = s.bill_debated_id
LEFT JOIN public.core_session cs ON cs.id = b.session_id
WHERE s.procedural = false
  AND s.bill_debated_id IS NOT NULL
  AND s.content_en IS NOT NULL
  AND (
        s.content_en ILIKE '%voted for%'
     OR s.content_en ILIKE '%voted against%'
  );
