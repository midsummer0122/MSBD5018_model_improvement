"""Offline regression checks for non-destructive imports and isolated config updates."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch

import import_parent_chunks_to_mysql as importer
import setup_mysql_local as setup


class MySQLSetupTests(unittest.TestCase):
    def test_real_local_documents_and_bidirectional_links(self):
        counts=[];ids=set();parents=set()
        for directory in ('json','json_cases'):
            ds,ps=importer.load_parent_rows(importer.ROOT_DIR/'knowledge_data'/directory)
            self.assertFalse(ids & {d['doc_id'] for d in ds});ids.update(d['doc_id'] for d in ds)
            self.assertFalse(parents & {p['parent_id'] for p in ps});parents.update(p['parent_id'] for p in ps)
            counts.append((len(ds),len(ps)))
        self.assertEqual(counts,[(128,511),(174,349)])

    def test_identical_upsert_is_allowed_but_conflicting_text_is_rejected(self):
        ds,ps=importer.load_parent_rows(importer.ROOT_DIR/'knowledge_data/json',limit=1)
        d,p=ds[0],ps[0]
        conn=MagicMock();cursor=conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.side_effect=[[d],[p]]
        self.assertEqual(importer.check_existing_rows(conn,[d],[p]),{'matching_documents':1,'matching_parents':1})
        changed={**p,'text':p['text']+'modified'}
        cursor.fetchall.side_effect=[[d],[p]]
        with self.assertRaisesRegex(ValueError,'ID/content conflicts'):
            importer.check_existing_rows(conn,[d],[changed])
        self.assertTrue(all(call.args[0].lstrip().startswith('SELECT') for call in cursor.execute.call_args_list))

    def test_config_update_changes_only_mysql_section(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'api_key').mkdir();path=root/'api_key/api_config.yaml'
            original='llm:\r\n  api_key: "synthetic-test-value"\r\nmysql:\r\n  host: localhost\r\n  password: ""\r\napp:\r\n  debug: false\r\n'
            path.write_bytes(original.encode())
            with patch.object(setup,'ROOT',root):
                setup.update_mysql_config({'host':'127.0.0.1','password':'synthetic-local-password','port':3306})
            new=yaml.safe_load(path.read_text())
            self.assertEqual(new['llm'],yaml.safe_load(original)['llm'])
            self.assertEqual(new['app'],yaml.safe_load(original)['app'])
            self.assertEqual(new['mysql']['host'],'127.0.0.1')


if __name__=='__main__':unittest.main()
