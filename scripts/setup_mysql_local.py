"""Interactive Windows setup. Run in a local elevated console; passwords are hidden and never logged.

Uses only a newly verified official installation under G:/MySQL. Existing data
is never initialized again, and existing root passwords are never changed.
"""
from __future__ import annotations

import argparse
import ctypes
import getpass
import hashlib
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.config import AppConfig
from import_parent_chunks_to_mysql import load_parent_rows,check_existing_rows,write_to_mysql
from verify_mysql_parent_store import verify_all

OUT=ROOT/'data/mysql_setup'


def status(phase,**values):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'setup_status.json').write_text(json.dumps({'phase':phase,**values},ensure_ascii=False,indent=2),encoding='utf-8')
    print(phase,flush=True)


def run_public(command):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace')
    # These commands never contain any password. Keep output local for service troubleshooting.
    with (OUT/'service_setup.log').open('a',encoding='utf-8') as f:
        f.write('COMMAND: '+' '.join(map(str,command))+'\n'+result.stdout+result.stderr+'\n')
    if result.returncode:raise RuntimeError('Native MySQL/service command failed; inspect service_setup.log')
    return result


def mysql_connect(port,password,user='root',database=None):
    import pymysql
    return pymysql.connect(host='127.0.0.1',port=port,user=user,password=password,database=database,
        charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor,connect_timeout=3,autocommit=True)


def check_schema(conn):
    required={'documents':{'id','doc_id','title','source_path','source_json','structure_type','file_hash'},
        'parent_chunks':{'id','document_id','parent_id','title_path','text','child_ids','parent_index','content_hash'}}
    with conn.cursor() as cursor:
        for table,columns in required.items():
            cursor.execute(f'SHOW COLUMNS FROM law_rag.{table}')
            info={r['Field']:r for r in cursor.fetchall()}
            if not columns<=set(info):raise ValueError(f'Existing table missing columns: {table}')
            key='doc_id' if table=='documents' else 'parent_id'
            if not any(t in info[key]['Type'].lower() for t in ('char','text')):raise ValueError(f'ID must be a string: {table}')
            if table=='parent_chunks' and 'longtext' not in info['text']['Type'].lower():
                raise ValueError('Parent text column must be LONGTEXT; inspect before altering existing data')
            cursor.execute(f'SHOW INDEX FROM law_rag.{table}')
            indexes=cursor.fetchall();groups={}
            for row in indexes:groups.setdefault(row['Key_name'],[]).append(row)
            if not any(len(rows)==1 and rows[0]['Column_name']==key and not rows[0]['Non_unique'] for rows in groups.values()):
                raise ValueError(f'Missing unique upsert index: {table}.{key}')


def update_mysql_config(section):
    import re,yaml
    path=ROOT/'api_key/api_config.yaml';raw=path.read_bytes();bom=raw.startswith(b'\xef\xbb\xbf')
    text=raw.decode('utf-8-sig');before=yaml.safe_load(text)
    start=re.search(r'(?m)^mysql:[^\r\n]*(?:\r?\n|$)',text)
    if not start:raise ValueError('Expected existing mysql configuration section')
    next_section=re.search(r'(?m)^[A-Za-z_][A-Za-z0-9_-]*:[^\r\n]*(?:\r?\n|$)',text[start.end():])
    end=start.end()+next_section.start() if next_section else len(text)
    replacement=yaml.safe_dump({'mysql':section},allow_unicode=True,sort_keys=False)
    if '\r\n' in text:replacement=replacement.replace('\n','\r\n')
    updated=text[:start.start()]+replacement+('\r\n' if '\r\n' in text else '\n')+text[end:]
    after=yaml.safe_load(updated)
    if {k:v for k,v in before.items() if k!='mysql'}!={k:v for k,v in after.items() if k!='mysql'}:
        raise AssertionError('Non-MySQL configuration would change')
    path.write_bytes((b'\xef\xbb\xbf' if bom else b'')+updated.encode('utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-dir',type=Path,default=Path('G:/MySQL/mysql-8.4.13-winx64'))
    parser.add_argument('--data-dir',type=Path,default=Path('G:/MySQL/law_rag_data'))
    parser.add_argument('--service',default='MySQL84LawRag');parser.add_argument('--port',type=int,default=3306)
    args=parser.parse_args()
    if not ctypes.windll.shell32.IsUserAnAdmin():raise PermissionError('An elevated local console is required')
    for path in (args.server_dir.resolve(),args.data_dir.resolve()):
        if not path.is_relative_to(Path('G:/MySQL').resolve()):raise ValueError('Installation and data must stay under G:/MySQL')
    mysqld=args.server_dir/'bin/mysqld.exe'
    if not mysqld.exists():raise FileNotFoundError('Official MySQL program is missing')
    initialized=(args.data_dir/'auto.cnf').exists() and (args.data_dir/'mysql.ibd').exists()
    if args.data_dir.exists() and any(args.data_dir.iterdir()) and not initialized:
        raise ValueError('Nonempty data directory is not a confirmed initialized instance; refusing initialization')
    service_query=subprocess.run(['sc.exe','query',args.service],capture_output=True)
    if service_query.returncode==0 and not initialized:raise ValueError('Existing service requires inspection before initializing')
    status('awaiting_local_password',fresh_install=not initialized,service=args.service,port=args.port)
    if initialized:
        root_password=getpass.getpass('Existing local MySQL root password (hidden): ')
    else:
        print('Set the NEW root password for this fresh local MySQL instance. It will NOT be saved by this setup script.')
        while True:
            root_password=getpass.getpass('New root password (hidden): ')
            repeat=getpass.getpass('Confirm new root password (hidden): ')
            if root_password and root_password==repeat:break
            print('Passwords must be nonempty and match. Please try again.')
        del repeat
    status('preparing_local_instance')
    logs=Path('G:/MySQL/logs');logs.mkdir(exist_ok=True)
    ini=Path('G:/MySQL/my-law-rag.ini')
    if not initialized:
        if ini.exists():raise ValueError('Existing MySQL option file requires inspection; refusing overwrite')
        args.data_dir.mkdir(exist_ok=True)
        ini.write_text('[mysqld]\n'+f'basedir={args.server_dir.as_posix()}\ndatadir={args.data_dir.as_posix()}\n'
            f'port={args.port}\nbind-address=127.0.0.1\nmysqlx=0\ncharacter-set-server=utf8mb4\n'
            'collation-server=utf8mb4_unicode_ci\nskip-log-bin\ngeneral-log=0\nslow-query-log=0\n'
            'log-error=G:/MySQL/logs/law-rag-error.log\n'
            f'[client]\nhost=127.0.0.1\nport={args.port}\ndefault-character-set=utf8mb4\n',encoding='utf-8')
        run_public([str(mysqld),f'--defaults-file={ini}','--validate-config'])
        run_public([str(mysqld),f'--defaults-file={ini}','--initialize-insecure'])
    if service_query.returncode!=0:
        run_public([str(mysqld),'--install',args.service,f'--defaults-file={ini}'])
    # Start only this project's service; never stop or modify another instance.
    subprocess.run(['sc.exe','start',args.service],capture_output=True)
    conn=None
    for attempt in range(40):
        try:
            conn=mysql_connect(args.port,root_password if initialized else '')
            break
        except Exception:
            time.sleep(0.5)
    if conn is None:raise ConnectionError('Local MySQL did not become available or credentials were rejected')
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT @@datadir AS data_dir')
            if Path(cursor.fetchone()['data_dir']).resolve()!=args.data_dir.resolve():
                raise ValueError('Connected server data directory differs from the intended local instance')
        if not initialized:
            with conn.cursor() as cursor:cursor.execute("ALTER USER 'root'@'localhost' IDENTIFIED BY %s",(root_password,))
        root_password=None
        status('creating_or_checking_schema')
        with conn.cursor() as cursor:
            sql='\n'.join(line for line in (ROOT/'scripts/mysql_schema.sql').read_text(encoding='utf-8').splitlines()
                          if not line.lstrip().startswith('--'))
            for statement in sql.split(';'):
                if statement.strip():cursor.execute(statement)
        check_schema(conn)
        documents=[];parents=[];counts={}
        for directory in ('json','json_cases'):
            d,p=load_parent_rows(ROOT/'knowledge_data'/directory);documents.extend(d);parents.extend(p)
            counts[directory]={'documents':len(d),'parents':len(p)}
        if len({d['doc_id'] for d in documents})!=len(documents) or len({p['parent_id'] for p in parents})!=len(parents):
            raise ValueError('Cross-directory ID collision')
        check_existing_rows(conn,documents,parents)
        previous=AppConfig.load().section('mysql')
        project_user=None;project_password=None
        if previous.get('user') and previous.get('password'):
            try:
                old=mysql_connect(args.port,previous['password'],previous['user'],'law_rag');old.close()
                project_user=previous['user'];project_password=previous['password']
            except Exception:pass
        if project_user is None:
            with conn.cursor() as cursor:
                base='law_rag_app';project_user=base
                for suffix in range(100):
                    candidate=base if suffix==0 else base+str(suffix)
                    cursor.execute('SELECT COUNT(*) AS n FROM mysql.user WHERE User=%s',(candidate,))
                    if cursor.fetchone()['n']==0:project_user=candidate;break
                else:raise ValueError('Cannot find an unused project account name')
                project_password=secrets.token_urlsafe(32)
                cursor.execute('CREATE USER %s@%s IDENTIFIED BY %s',(project_user,'127.0.0.1',project_password))
                cursor.execute('GRANT SELECT, INSERT, UPDATE ON law_rag.* TO %s@%s',(project_user,'127.0.0.1'))
        status('importing_parent_chunks',counts=counts,project_user=project_user)
        for directory in ('json','json_cases'):
            d,p=load_parent_rows(ROOT/'knowledge_data'/directory)
            import_args=argparse.Namespace(host='127.0.0.1',port=args.port,user=project_user,password=project_password,
                password_env=None,ask_password=False,database='law_rag',commit_every=20)
            write_to_mysql(import_args,d,p)
        mysql_section={'host':'127.0.0.1','port':args.port,'user':project_user,'password':project_password,
                       'database':'law_rag','charset':'utf8mb4'}
        config=AppConfig.load();config.raw['mysql']=mysql_section
        status('verifying_all_mysql_parents')
        verification=verify_all(config)
        update_mysql_config(mysql_section)
        # Reload the actual application YAML, proving the persisted configuration works too.
        verification=verify_all(AppConfig.load())
        hashes=json.loads((OUT/'non_mysql_config_hashes.json').read_text(encoding='utf-8'))
        current=AppConfig.load().raw
        assert all(hashlib.sha256(json.dumps(current[k],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==v for k,v in hashes.items())
        (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
        status('complete',service=args.service,port=args.port,server_dir=str(args.server_dir),
               data_dir=str(args.data_dir),option_file=str(ini),project_user=project_user,
               database='law_rag',counts=counts,verification=verification,non_mysql_config_unchanged=True,
               root_password_saved=False)
        print('MySQL setup and all-parent verification passed. Root password was not saved. The application account password is only in the local YAML.')
    finally:conn.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,mysql_errno=exc.args[0] if exc.args and isinstance(exc.args[0],int) else None)
        print('Setup failed. No passwords are shown. Ask the assistant to inspect the sanitized status and local service log.')
        raise SystemExit(1)
