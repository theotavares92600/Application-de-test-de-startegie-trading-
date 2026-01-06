# Configuration de l'Automatisation (Cron Job)

Ce projet utilise **Cron** (le planificateur de tâches Linux) pour générer automatiquement un rapport financier quotidien.

## 1. Fréquence
Le rapport est généré tous les jours à **20h00**.

## 2. Commande configurée
Voici la ligne ajoutée au fichier crontab de la machine virtuelle (`crontab -e`) :

0 20 * * * /home/geheres/projet_finance/venv/bin/python /home/geheres/projet_finance/report.py >> /home/geheres/projet_finance/cron_log.txt 2>&1
