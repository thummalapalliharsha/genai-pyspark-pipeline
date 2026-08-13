from src.spark_analytics import create_spark_session


def test_create_spark_session():
    """Ensure `create_spark_session` returns an object with `sparkContext.appName` and `stop()`.

    This test accepts either a real SparkSession (when Java is available) or the
    lightweight fallback mock implemented for environments without a JVM.
    """
    spark = create_spark_session(app_name="test_session")
    assert spark is not None
    assert hasattr(spark, "sparkContext") and hasattr(spark.sparkContext, "appName")
    assert spark.sparkContext.appName == "test_session"
    # stop should be callable
    assert callable(getattr(spark, "stop"))
    spark.stop()
