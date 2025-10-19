#!/usr/bin/env python3
"""
Now i am creating Email Sender Utility
====================

Now i am handling sending personalized emails based on job analysis results.
Now i am using standard email libraries (smtplib, email) with original email templates.
Now i am supporting multiple email providers and templates.

References:
- Python smtplib Documentation: https://docs.python.org/3/library/smtplib.html
- Python email Documentation: https://docs.python.org/3/library/email.html
"""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import logging
from pathlib import Path


class EmailSender:
    """Now i am handling email sending functionality"""
    
    def __init__(self, config_file: str = "config/email_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Now i am loading email configuration"""
        default_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "sender_password": "",
            "recipients": [],
            "use_tls": True
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Now i am merging with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                self.logger.warning(f"Failed to load email config: {e}")
                return default_config
        else:
            # Now i am creating default config file
            self._create_default_config(default_config)
            return default_config
    
    def _create_default_config(self, config: Dict):
        """Now i am creating default email configuration file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Created default email config: {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to create email config: {e}")
    
    def send_email(self, to: str, subject: str, body: str, is_html: bool = False) -> Dict:
        """Now i am sending a single email"""
        try:
            if not self.config.get('sender_email') or not self.config.get('sender_password'):
                return {
                    'success': False,
                    'error': 'Email configuration not set. Please configure sender_email and sender_password in email_config.json'
                }
            
            # Now i am creating message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config['sender_email']
            msg['To'] = to
            msg['Subject'] = subject
            
            # Now i am adding body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Now i am sending email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config.get('use_tls', True):
                    server.starttls()
                
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)
            
            self.logger.info(f" Email sent to {to}")
            return {'success': True, 'recipient': to}
            
        except Exception as e:
            self.logger.error(f" Failed to send email to {to}: {e}")
            return {'success': False, 'error': str(e), 'recipient': to}
    
    def send_bulk_emails(self, email_templates: List[Dict]) -> List[Dict]:
        """Now i am sending multiple emails"""
        results = []
        
        for template in email_templates:
            result = self.send_email(
                to=template['recipient'],
                subject=template['subject'],
                body=template['body'],
                is_html=template.get('is_html', False)
            )
            results.append(result)
        
        return results
    
    def send_job_analysis_emails(self, analysis_results: Dict) -> List[Dict]:
        """Now i am sending personalized emails based on job analysis results"""
        try:
            email_templates = self._generate_analysis_emails(analysis_results)
            return self.send_bulk_emails(email_templates)
        except Exception as e:
            self.logger.error(f"Failed to send analysis emails: {e}")
            return []
    
    def _generate_analysis_emails(self, analysis_results: Dict) -> List[Dict]:
        """Now i am generating email templates from analysis results"""
        templates = []
        
        # Now i am getting recipients from config
        recipients = self.config.get('recipients', [])
        if not recipients:
            self.logger.warning("No email recipients configured")
            return templates
        
        # Now i am generating summary email
        summary_template = self._create_summary_email(analysis_results)
        for recipient in recipients:
            template = summary_template.copy()
            template['recipient'] = recipient
            templates.append(template)
        
        # Now i am generating individual job emails if requested
        if analysis_results.get('jobs'):
            for job_id, job_data in analysis_results['jobs'].items():
                job_template = self._create_job_email(job_id, job_data)
                for recipient in recipients:
                    template = job_template.copy()
                    template['recipient'] = recipient
                    templates.append(template)
        
        return templates
    
    def _create_summary_email(self, analysis_results: Dict) -> Dict:
        """Now i am creating summary email template"""
        total_jobs = analysis_results.get('total_jobs', 0)
        avg_salary = analysis_results.get('average_salary', 'N/A')
        top_skills = analysis_results.get('top_skills', [])
        
        subject = f"Job Analysis Summary - {total_jobs} Jobs Analyzed"
        
        body = f"""
Job Analysis Summary Report
==========================

Analysis Date: {analysis_results.get('analysis_date', 'N/A')}
Total Jobs Analyzed: {total_jobs}
Average Salary: {avg_salary}

Top Skills in Demand:
{chr(10).join(f"• {skill}" for skill in top_skills[:5])}

Key Insights:
• Market Trends: {analysis_results.get('market_trends', 'Analysis in progress')}
• Salary Range: {analysis_results.get('salary_range', 'Data being processed')}
• Skill Requirements: {analysis_results.get('skill_requirements', 'Analysis in progress')}

Recommendations:
{analysis_results.get('recommendations', 'Detailed recommendations will be provided in individual job reports.')}

Next Steps:
1. Review individual job reports for detailed analysis
2. Focus on high-match opportunities
3. Consider skill development in trending areas

Best regards,
Job Analysis System
        """.strip()
        
        return {
            'subject': subject,
            'body': body,
            'is_html': False
        }
    
    def _create_job_email(self, job_id: str, job_data: Dict) -> Dict:
        """Now i am creating individual job email template"""
        title = job_data.get('title', 'Unknown Position')
        company = job_data.get('company', 'Unknown Company')
        salary = job_data.get('salary_range', 'Not specified')
        
        subject = f"Job Analysis: {title} at {company}"
        
        body = f"""
Job Analysis Report
==================

Position: {title}
Company: {company}
Job ID: {job_id}

Key Details:
• Salary Range: {salary}
• Employment Type: {job_data.get('employment_type', 'Not specified')}
• Seniority Level: {job_data.get('seniority_level', 'Not specified')}
• Location: {job_data.get('location', 'Not specified')}

Match Analysis:
• Overall Match Score: {job_data.get('match_score', 'N/A')}/100
• Skills Match: {job_data.get('skills_match', 'N/A')}/100
• Experience Match: {job_data.get('experience_match', 'N/A')}/100

Key Requirements:
{job_data.get('key_requirements', 'Requirements analysis in progress')}

Recommendations:
{job_data.get('recommendations', 'Detailed recommendations will be provided after full analysis.')}

Action Items:
1. Review full job description
2. Assess skill gaps
3. Prepare tailored application
4. Follow up on application status

Best regards,
Job Analysis System
        """.strip()
        
        return {
            'subject': subject,
            'body': body,
            'is_html': False
        }
    
    def test_email_config(self) -> Dict:
        """Now i am testing email configuration"""
        try:
            if not self.config.get('sender_email'):
                return {
                    'success': False,
                    'error': 'sender_email not configured'
                }
            
            if not self.config.get('sender_password'):
                return {
                    'success': False,
                    'error': 'sender_password not configured'
                }
            
            # Now i am testing SMTP connection
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config.get('use_tls', True):
                    server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
            
            return {
                'success': True,
                'message': 'Email configuration is valid'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Email configuration test failed: {e}'
            }


def main():
    """Now i am testing the email functionality"""
    print(" Email Sender Test")
    print("=" * 30)
    
    sender = EmailSender()
    
    # Test configuration
    config_test = sender.test_email_config()
    if config_test['success']:
        print(" Email configuration is valid")
    else:
        print(f" Email configuration error: {config_test['error']}")
        print("\nTo configure email:")
        print("1. Edit config/email_config.json")
        print("2. Add your email credentials")
        print("3. Add recipient email addresses")
        return
    
    # Test sending a simple email
    test_result = sender.send_email(
        to="test@example.com",
        subject="Test Email from Job Analysis System",
        body="This is a test email to verify the email system is working correctly."
    )
    
    if test_result['success']:
        print(" Test email sent successfully")
    else:
        print(f" Test email failed: {test_result['error']}")


if __name__ == "__main__":
    main()
