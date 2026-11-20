-- Compatible with import_parent_chunks_to_mysql.py and ParentChunkStore.
-- Existing databases/tables are preserved; inspect their structure before import.
CREATE DATABASE IF NOT EXISTS law_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE law_rag;

CREATE TABLE IF NOT EXISTS documents (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    doc_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    title TEXT NULL,
    source_path TEXT NULL,
    source_json VARCHAR(512) NULL,
    structure_type VARCHAR(64) NULL,
    file_hash CHAR(64) CHARACTER SET ascii NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_documents_doc_id (doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS parent_chunks (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    document_id BIGINT UNSIGNED NOT NULL,
    parent_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    title_path TEXT NULL,
    text LONGTEXT NOT NULL,
    child_ids LONGTEXT NULL,
    parent_index INT UNSIGNED NOT NULL,
    content_hash CHAR(64) CHARACTER SET ascii NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_parent_chunks_parent_id (parent_id),
    KEY idx_parent_chunks_document_id (document_id),
    CONSTRAINT fk_parent_chunks_document FOREIGN KEY (document_id)
        REFERENCES documents(id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
