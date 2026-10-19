"""Compõe artigo de dez páginas a partir de artigo/conteudo.json e equipe.json.

Altere os textos em conteudo.json e a identificação em equipe.json; execute
python -m src.build_article. O programa recusa páginas que excedam a área útil.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]

def register_fonts():
    options = [Path('/usr/share/fonts/truetype/liberation'), Path('/usr/share/fonts/truetype/liberation2')]
    for folder in options:
        if (folder / 'LiberationSerif-Regular.ttf').exists():
            for suffix, fn in [('','Regular'),('-Bold','Bold'),('-Italic','Italic'),('-BoldItalic','BoldItalic')]:
                pdfmetrics.registerFont(TTFont('Article'+suffix, str(folder/f'LiberationSerif-{fn}.ttf')))
            pdfmetrics.registerFontFamily('Article',normal='Article',bold='Article-Bold',italic='Article-Italic',boldItalic='Article-BoldItalic')
            return 'Article'
    # Fontes-base PDF mantêm o gerador portátil; Linux usado no estudo embute Liberation.
    return 'Times-Roman'

def build(root):
    team=json.loads((root/'artigo/equipe.json').read_text())
    document=json.loads((root/'artigo/conteudo.json').read_text())
    if len(document['pages']) > 10:
        raise ValueError('O artigo não pode exceder dez páginas.')
    font=register_fonts()
    bold='Article-Bold' if font=='Article' else 'Times-Bold'
    base=dict(fontName=font,fontSize=12,leading=14,alignment=TA_JUSTIFY,spaceAfter=7)
    styles={
        'body':ParagraphStyle('body',**base),
        'small':ParagraphStyle('small',fontName=font,fontSize=11,leading=12.5,alignment=TA_JUSTIFY,spaceAfter=6),
        'caption':ParagraphStyle('caption',fontName=font,fontSize=10,leading=11.5,alignment=TA_LEFT,spaceAfter=4),
        'source':ParagraphStyle('source',fontName=font,fontSize=9,leading=10.5,alignment=TA_LEFT,spaceAfter=7),
        'h1':ParagraphStyle('h1',fontName=bold,fontSize=12,leading=14,alignment=TA_LEFT,spaceAfter=9),
        'h2':ParagraphStyle('h2',fontName=bold,fontSize=12,leading=14,alignment=TA_LEFT,spaceAfter=7),
        'title':ParagraphStyle('title',fontName=bold,fontSize=14,leading=16,alignment=TA_CENTER,spaceAfter=10),
        'authors':ParagraphStyle('authors',fontName=font,fontSize=10,leading=12,alignment=TA_CENTER,spaceAfter=6),
        'reference':ParagraphStyle('reference',fontName=font,fontSize=11,leading=12.5,alignment=TA_LEFT,spaceAfter=6,splitLongWords=True),
    }
    out=root/'artigo/artigo_redes_neurais.pdf'
    out.parent.mkdir(exist_ok=True)
    c=canvas.Canvas(str(out), pagesize=A4,pageCompression=1)
    c.setTitle(team['titulo'])
    c.setAuthor('; '.join(a['nome'] for a in team['autores']))
    c.setSubject('Estudo experimental do TrashNet: seis treinos, dados, métricas e custos verificáveis')
    width,height=A4
    left,right,top,bottom=3*cm,2*cm,2.6*cm,2*cm
    usable=width-left-right
    markdown=[]
    diagnostics=[]
    for page_index, blocks in enumerate(document['pages'],1):
        y=height-top
        c.setFont(font,10)
        c.drawRightString(width-right,height-1.55*cm,str(page_index))
        for block_index, block in enumerate(blocks):
            kind=block['type']
            if kind=='identification':
                items=[('title',html.escape(team['titulo'])),
                       ('authors','<br/>'.join(html.escape(a['nome'])+' - matrícula '+html.escape(a['matricula']) for a in team['autores'])),
                       ('authors','Atividade final - Redes Neurais Artificiais'),
                       ('small','<b>Repositório GitHub:</b> '+(f'<link href="{html.escape(team["github_url"],quote=True)}">{html.escape(team["github_url"])}</link>' if team.get('github_url') else 'publicação pendente; inserir o endereço público antes da submissão.'))]
                if not team.get('revisao_da_equipe_concluida'):
                    items.append(('small','<b>Versão para revisão da equipe.</b> Matrículas provisórias informadas pelos autores.'))
                flowables=[(Paragraph(t,styles[s]),styles[s].spaceAfter) for s,t in items]
                markdown.extend(re.sub('<[^>]+>','',t.replace('<br/>','\n')) for _,t in items)
            elif kind in styles:
                txt=block['text']
                flowables=[(Paragraph(txt,styles[kind]), block.get('after',styles[kind].spaceAfter))]
                markdown.append(('## ' if kind in ('h1','h2') else '')+re.sub('<[^>]+>','',txt))
            elif kind=='space':
                y-=block.get('points',5); continue
            elif kind=='figure':
                path=root/block['path']
                image_width=usable*block.get('width_fraction',1)
                with PILImage.open(path) as im:
                    image_height=image_width*im.height/im.width
                if block.get('max_height_cm') and image_height>block['max_height_cm']*cm:
                    ratio=block['max_height_cm']*cm/image_height
                    image_width*=ratio;image_height*=ratio
                flowables=[(Image(str(path),width=image_width,height=image_height),block.get('after',3))]
                markdown.append(f'![{block.get("alt",path.stem)}](../{block["path"]})')
            elif kind=='table':
                fontsize=block.get('font_size',10)
                cellstyle=ParagraphStyle('cell',fontName=font,fontSize=fontsize,leading=fontsize+1.5,alignment=TA_LEFT)
                rows=[[Paragraph(str(v),cellstyle) for v in row] for row in block['rows']]
                fractions=block.get('widths',[1/len(rows[0])]*len(rows[0]))
                table=Table(rows,colWidths=[usable*x for x in fractions],hAlign='LEFT')
                settings=[('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),
                          ('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),block.get('padding',3)),
                          ('BOTTOMPADDING',(0,0),(-1,-1),block.get('padding',3)),
                          ('LINEABOVE',(0,0),(-1,0),.8,colors.black),
                          ('LINEBELOW',(0,0),(-1,0),.5,colors.black),
                          ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),
                          ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee'))]
                for row in block.get('separators',[]): settings.append(('LINEABOVE',(0,row),(-1,row),.3,colors.HexColor('#b0b0b0')))
                for span in block.get('spans',[]): settings.append(('SPAN',tuple(span[0]),tuple(span[1])))
                table.setStyle(TableStyle(settings))
                flowables=[(table,block.get('after',6))]
                rendered_rows=['| '+' | '.join(re.sub('<[^>]+>','',str(v)) for v in row)+' |' for row in block['rows']]
                rendered_rows.insert(1,'| '+' | '.join('---' for _ in block['rows'][0])+' |')
                markdown.extend(rendered_rows)
            else:
                raise ValueError(f'Tipo desconhecido: {kind}')
            for flowable,after in flowables:
                w,h=flowable.wrap(usable,y-bottom)
                if h>y-bottom+.1:
                    raise ValueError(f'Página {page_index}, bloco {block_index} ({kind}): falta {h-(y-bottom):.1f} pt; ajuste a fonte de conteúdo, sem ocultar texto.')
                x=left+(usable-w)/2 if kind=='figure' else left
                flowable.drawOn(c,x,y-h)
                y-=h+after
            markdown.append('')
        diagnostics.append({'page':page_index,'remaining_height_pt':round(y-bottom,2)})
        c.showPage()
        markdown.append('\n---\n')
    c.save()
    (root/'artigo/artigo_editavel.md').write_text('\n'.join(markdown),encoding='utf-8')
    (root/'artigo/layout_check.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps({'pdf':str(out),'pages':len(document['pages']),'layout':diagnostics},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();build(args.root.resolve())
