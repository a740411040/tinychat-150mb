import copy
import unittest

from tinychat.natural_sft_data import audit_cmrc


class GroundedEvidenceTests(unittest.TestCase):
    def row(self):
        context='  小舟住在北京。  '
        return {'subtype':'quote','source_evidence':{
            'context':context,'question':'小舟住在哪里？','gold_answer':'北京',
            'gold_start':context.index('北京'),'passage_start':0,'passage_end':len(context),
            'evidence_sentence_start':0,'evidence_sentence_end':len(context)},
            'messages':[{'role':'user','content':'请根据材料回答。\n\n小舟住在北京。\n\n小舟住在哪里？'},
                        {'role':'assistant','content':'北京\n原文写道：“小舟住在北京。”'}]}

    def test_rendered_whitespace_is_allowed(self):
        audit_cmrc(self.row())

    def test_unsupported_suffix_is_rejected(self):
        row=self.row()
        row['messages'][1]['content']+='他后来搬去了上海。'
        with self.assertRaisesRegex(ValueError,'complete response'):
            audit_cmrc(row)

    def test_changed_question_is_rejected(self):
        row=self.row()
        row['messages'][0]['content']=row['messages'][0]['content'].replace('小舟住在哪里？','小舟几岁？')
        with self.assertRaisesRegex(ValueError,'visible passage'):
            audit_cmrc(row)

    def test_gold_outside_visible_passage_is_rejected(self):
        row=copy.deepcopy(self.row())
        row['source_evidence']['passage_end']=row['source_evidence']['gold_start']
        with self.assertRaisesRegex(ValueError,'excludes answer'):
            audit_cmrc(row)


if __name__=='__main__':
    unittest.main()
