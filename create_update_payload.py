import json, base64

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernel_spec": {"display_name": "PySpark", "language": "python", "name": "pysparkkernel"},
        "language_info": {"name": "python"}
    },
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pyspark.sql import SparkSession\n",
                "from pyspark.sql.types import (\n",
                "    StructType, StructField, StringType, DoubleType, BooleanType\n",
                ")\n",
                "from datetime import datetime\n",
                "spark = SparkSession.builder.getOrCreate()\n",
                "print('Spark initialized.')\n",
                "\n",
                "price_schema = StructType([\n",
                "    StructField('timestamp', StringType(), False),\n",
                "    StructField('metal', StringType(), False),\n",
                "    StructField('price', DoubleType(), False),\n",
                "    StructField('unit', StringType(), True),\n",
                "    StructField('source', StringType(), True),\n",
                "    StructField('change_pct', DoubleType(), True),\n",
                "])\n",
                "initial_prices = [\n",
                "    (datetime.utcnow().isoformat(), 'Nickel', 18500.0, 'USD/mt', 'AlphaVantage', 0.0),\n",
                "    (datetime.utcnow().isoformat(), 'Cobalt', 32000.0, 'USD/mt', 'AlphaVantage', 0.0),\n",
                "    (datetime.utcnow().isoformat(), 'Lithium', 25000.0, 'USD/mt', 'AlphaVantage', 0.0),\n",
                "    (datetime.utcnow().isoformat(), 'Copper', 8500.0, 'USD/mt', 'AlphaVantage', 0.0),\n",
                "    (datetime.utcnow().isoformat(), 'Manganese', 3200.0, 'USD/mt', 'AlphaVantage', 0.0),\n",
                "]\n",
                "df = spark.createDataFrame(initial_prices, price_schema)\n",
                "df.write.format('delta').mode('overwrite').saveAsTable('CommodityPrices')\n",
                "print('CommodityPrices: created')\n",
                "\n",
                "calc_schema = StructType([\n",
                "    StructField('timestamp', StringType(), False),\n",
                "    StructField('contract_type', StringType(), False),\n",
                "    StructField('values_json', StringType(), False),\n",
                "])\n",
                "df2 = spark.createDataFrame([(datetime.utcnow().isoformat(), 'seed', '{}')], calc_schema)\n",
                "df2.write.format('delta').mode('overwrite').saveAsTable('ContractCalculations')\n",
                "print('ContractCalculations: created')\n",
                "\n",
                "analysis_schema = StructType([\n",
                "    StructField('timestamp', StringType(), False),\n",
                "    StructField('ni_price', DoubleType(), False),\n",
                "    StructField('co_price', DoubleType(), False),\n",
                "    StructField('li_price', DoubleType(), False),\n",
                "    StructField('black_mass_value', DoubleType(), True),\n",
                "    StructField('mhp_profit_share_triggered', BooleanType(), True),\n",
                "    StructField('li_floor_active', BooleanType(), True),\n",
                "    StructField('li_ceiling_active', BooleanType(), True),\n",
                "    StructField('ai_analysis', StringType(), True),\n",
                "])\n",
                "df3 = spark.createDataFrame([(datetime.utcnow().isoformat(), 0.0, 0.0, 0.0, None, None, None, None, 'seed')], analysis_schema)\n",
                "df3.write.format('delta').mode('overwrite').saveAsTable('ContractAnalysis')\n",
                "print('ContractAnalysis: created')\n",
                "\n",
                "reg_schema = StructType([\n",
                "    StructField('timestamp', StringType(), False),\n",
                "    StructField('regulation_id', StringType(), True),\n",
                "    StructField('title', StringType(), True),\n",
                "    StructField('agency', StringType(), True),\n",
                "    StructField('relevance_score', DoubleType(), True),\n",
                "    StructField('summary', StringType(), True),\n",
                "])\n",
                "df4 = spark.createDataFrame([(datetime.utcnow().isoformat(), None, None, None, None, 'seed')], reg_schema)\n",
                "df4.write.format('delta').mode('overwrite').saveAsTable('RegulatoryAlerts')\n",
                "print('RegulatoryAlerts: created')\n",
                "\n",
                "tables = spark.catalog.listTables()\n",
                "print('\\n=== All Lakehouse Tables ===')\n",
                "for t in tables:\n",
                "    print(f'  {t.name}: {spark.table(t.name).count()} records')\n",
                "print('All 4 Delta tables created!')"
            ]
        }
    ]
}

notebook_b64 = base64.b64encode(json.dumps(notebook).encode()).decode()

# Save plain notebook update (no .platform)
update_no_platform = {
    "definition": {
        "format": "ipynb",
        "parts": [
            {
                "path": "notebook.ipynb",
                "payload": notebook_b64,
                "payloadType": "InlineBase64"
            }
        ]
    }
}

with open('***REDACTED_PATH***/update_no_platform.json', 'w') as f:
    json.dump(update_no_platform, f)

print("Created update_no_platform.json")
