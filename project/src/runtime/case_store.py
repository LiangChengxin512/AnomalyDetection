"""Transactional, idempotent operator workflow and append-only audit exports."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from src.common.io_utils import json_safe
from src.common.provenance import now,digest


class CaseStore:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path/'cases.sqlite')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY, alarm_id TEXT UNIQUE, state TEXT, recovery_observed INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS actions(action_id TEXT PRIMARY KEY, case_id TEXT, action TEXT, actor TEXT, timestamp TEXT, reason TEXT, from_state TEXT, to_state TEXT);
        CREATE TABLE IF NOT EXISTS frames(sample_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS session(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        ''')
    def bind(self,identity):
        previous=self.db.execute("SELECT value FROM session WHERE key='identity'").fetchone()
        encoded=json.dumps(identity,sort_keys=True)
        if previous and previous[0]!=encoded:raise ValueError('Session bundle or source changed; choose a new session ID')
        with self.db:self.db.execute("INSERT OR IGNORE INTO session VALUES('identity',?)",(encoded,))
    def contains(self,sample_id):
        return bool(self.db.execute('SELECT 1 FROM frames WHERE sample_id=?',(sample_id,)).fetchone())
    def apply_update(self,update):
        sample=update['detection']['sample_id']
        if self.contains(sample):return False
        alarm=update['alarm'];case=alarm.get('alarm_id')
        with self.db:
            self.db.execute('INSERT INTO frames VALUES(?,?)',(sample,json.dumps(json_safe(update),allow_nan=False)))
            if case:
                self.db.execute("INSERT OR IGNORE INTO cases(case_id,alarm_id,state) VALUES(?,?,'OPEN')",(case,case))
                if alarm['closed']:
                    self.db.execute('UPDATE cases SET recovery_observed=1 WHERE case_id=?',(case,))
                    self._action(case,'recovery-observed','system','Historical replay recovery observed; no causal claim',None)
        return True
    def _action(self,case,command,actor,reason,new_state):
        row=self.db.execute('SELECT state FROM cases WHERE case_id=?',(case,)).fetchone()
        if not row:raise ValueError(f'Unknown case: {case}')
        old=row[0];new=new_state or old
        action_id=digest([case,command,actor,reason,now()])
        self.db.execute('INSERT INTO actions VALUES(?,?,?,?,?,?,?,?)',(action_id,case,command,actor,now(),reason,old,new))
        self.db.execute('UPDATE cases SET state=? WHERE case_id=?',(new,case))
    def command(self,case,command,actor,reason):
        if not actor.strip() or not reason.strip():raise ValueError('Actor and reason are required')
        states={'acknowledge':({'OPEN'},'ACKNOWLEDGED'),'reject-advice':({'OPEN','ACKNOWLEDGED','UNDER_REVIEW'},'ADVICE_REJECTED'),
                'escalate':({'OPEN','ACKNOWLEDGED','UNDER_REVIEW','ADVICE_REJECTED'},'ESCALATED'),
                'add-note':({'OPEN','ACKNOWLEDGED','ADVICE_REJECTED','ESCALATED','UNDER_REVIEW'},'UNDER_REVIEW'),
                'close-case':({'ACKNOWLEDGED','UNDER_REVIEW','ADVICE_REJECTED','ESCALATED','RESOLVED'},'CLOSED')}
        if command not in states:raise ValueError(f'Unknown case command: {command}')
        row=self.db.execute('SELECT state FROM cases WHERE case_id=?',(case,)).fetchone()
        if not row or row[0] not in states[command][0]:raise ValueError('Invalid case transition; acknowledge or review the open case first')
        with self.db:
            if command=='close-case':self._action(case,'resolve',actor,reason,'RESOLVED')
            self._action(case,command,actor,reason,states[command][1])
    def export(self):
        for table,name in [('cases','cases.csv'),('actions','case_actions.csv')]:
            frame=pd.read_sql_query(f'SELECT * FROM {table}',self.db);frame.insert(0,'schema_version','2.0');frame.to_csv(self.path/name,index=False)
        with (self.path/'audit.jsonl').open('w',encoding='utf-8') as handle:
            for (payload,) in self.db.execute('SELECT payload FROM frames ORDER BY rowid'):handle.write(payload+'\n')
        return self.path
    def close(self):
        self.export();self.db.close()
