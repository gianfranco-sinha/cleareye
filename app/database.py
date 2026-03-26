"""Database connectivity and health checks for InfluxDB.

Supports both InfluxDB 1.x and 2.x.  Adapted from iaq4j for the ClearEye
water quality domain.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

# Import appropriate client based on version
try:
    from influxdb import InfluxDBClient

    HAS_INFLUXDB_V1 = True
except ImportError:
    InfluxDBClient = None  # type: ignore[assignment,misc]
    HAS_INFLUXDB_V1 = False

try:
    from influxdb_client import InfluxDBClient as InfluxDBClientV2

    HAS_INFLUXDB_V2 = True
except ImportError:
    InfluxDBClientV2 = None  # type: ignore[assignment,misc]
    HAS_INFLUXDB_V2 = False

logger = logging.getLogger(__name__)


class InfluxDBManager:
    """Manage InfluxDB connections with health checks.

    Reads connection parameters from ``database_config.yaml`` via
    :pymethod:`app.config.Settings.get_database_config`.
    """

    def __init__(self) -> None:
        self.client: Any = None
        self.client_type: str | None = None
        self.connected: bool = False
        self.last_error: str | None = None

        self.db_config = settings.get_database_config()
        self.version: str = self.db_config.get("version", "1.x")

        if self.db_config.get("enabled", False):
            try:
                self._connect()
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                logger.warning("InfluxDB unavailable at startup: %s", e)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        from app.exceptions import InfluxUnreachableError

        try:
            if self.version == "2.x":
                self._connect_v2()
            else:
                self._connect_v1()

            if self.connected:
                logger.info(
                    "Connected to InfluxDB %s at %s:%s",
                    self.version,
                    self.db_config.get("host"),
                    self.db_config.get("port"),
                )
            return self.connected
        except InfluxUnreachableError:
            raise
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            raise InfluxUnreachableError(
                f"Failed to connect to InfluxDB: {e}",
                suggestion="Check host/port/network",
            ) from e

    def _connect_v1(self) -> bool:
        from app.exceptions import InfluxUnreachableError

        if not HAS_INFLUXDB_V1:
            raise InfluxUnreachableError(
                "InfluxDB 1.x client not available",
                suggestion="Install influxdb package",
            )

        self.client = InfluxDBClient(
            host=self.db_config.get("host"),
            port=self.db_config.get("port"),
            username=self.db_config.get("username"),
            password=self.db_config.get("password"),
            database=self.db_config.get("database"),
            timeout=self.db_config.get("timeout", 60),
        )
        self.client.ping()

        databases = self.client.get_list_database()
        db_names = [db["name"] for db in databases]
        database_name = self.db_config.get("database")

        if database_name not in db_names:
            logger.warning(
                "Database '%s' not found. Available: %s", database_name, db_names
            )
            self.last_error = f"Database '{database_name}' does not exist"
            self.connected = False
            return False

        self.connected = True
        self.client_type = "InfluxDBClient"
        return True

    def _connect_v2(self) -> bool:
        from app.exceptions import InfluxUnreachableError

        if not HAS_INFLUXDB_V2:
            logger.warning(
                "InfluxDB 2.x client not available, falling back to 1.x client"
            )
            return self._connect_v1()

        token = self.db_config.get("token", "")
        org = self.db_config.get("org", "")

        if not token:
            raise InfluxUnreachableError(
                "Token is required for InfluxDB 2.x",
                suggestion="Set token in database_config.yaml",
            )
        if not org:
            raise InfluxUnreachableError(
                "Organization is required for InfluxDB 2.x",
                suggestion="Set org in database_config.yaml",
            )

        self.client = InfluxDBClientV2(
            url=f"http://{self.db_config.get('host')}:{self.db_config.get('port')}",
            token=token,
            org=org,
            timeout=self.db_config.get("timeout", 60) * 1000,
        )

        health = self.client.health()
        if health.status == "pass":
            self.connected = True
            self.client_type = "InfluxDBClientV2"
            return True

        self.last_error = f"InfluxDB 2.x health check failed: {health.message}"
        self.connected = False
        return False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        if not self.db_config.get("enabled", False):
            return {
                "enabled": False,
                "status": "disabled",
                "message": "InfluxDB integration is disabled",
            }

        if not self.connected:
            try:
                self._connect()
            except Exception as e:
                self.connected = False
                self.last_error = str(e)

        if self.connected:
            try:
                if self.version == "2.x":
                    health = self.client.health()
                    return {
                        "enabled": True,
                        "status": "healthy" if health.status == "pass" else "unhealthy",
                        "connected": True,
                        "host": self.db_config.get("host"),
                        "port": self.db_config.get("port"),
                        "database": self.db_config.get("bucket") or self.db_config.get("database"),
                        "version": self.version,
                        "client_type": self.client_type,
                    }
                else:
                    self.client.ping()
                    return {
                        "enabled": True,
                        "status": "healthy",
                        "connected": True,
                        "host": self.db_config.get("host"),
                        "port": self.db_config.get("port"),
                        "database": self.db_config.get("database"),
                        "version": self.version,
                        "client_type": self.client_type,
                    }
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                logger.error("InfluxDB health check failed: %s", e)

        return {
            "enabled": True,
            "status": "unhealthy",
            "connected": False,
            "host": self.db_config.get("host"),
            "port": self.db_config.get("port"),
            "database": self.db_config.get("database"),
            "version": self.version,
            "client_type": self.client_type,
            "error": self.last_error,
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        if not self.db_config.get("enabled", False):
            return False
        if not self.connected:
            try:
                self._connect()
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
        return self.connected

    def write_prediction(
        self,
        timestamp: float,
        turbidity_ntu: float,
        regime: str,
        calibration_method: str,
        rig_id: str | None = None,
        readings: dict[str, float] | None = None,
        biofouling_factor: float = 1.0,
        confidence: float | None = None,
    ) -> bool:
        """Write a calibrated prediction to InfluxDB.

        Args:
            timestamp: Unix timestamp.
            turbidity_ntu: Calibrated turbidity in NTU.
            regime: Turbidity regime (solution/colloid/suspension).
            calibration_method: Calibration method used.
            rig_id: Optional sensor rig identifier.
            readings: Optional raw sensor readings dict.
            biofouling_factor: Biofouling correction factor applied.
            confidence: Overall prediction confidence.
        """
        if not self._ensure_connected():
            logger.debug("InfluxDB not available — skipping write")
            return False

        measurement = self.db_config.get(
            "predictions_measurement", "calibrated_predictions"
        )

        try:
            fields: dict[str, float] = {"turbidity_ntu": float(turbidity_ntu)}
            if readings:
                fields.update({k: float(v) for k, v in readings.items()})
            fields["biofouling_factor"] = float(biofouling_factor)
            if confidence is not None:
                fields["confidence"] = float(confidence)

            tags: dict[str, str] = {
                "regime": regime,
                "calibration_method": calibration_method,
            }
            if rig_id:
                tags["rig_id"] = rig_id

            if self.version == "2.x":
                from datetime import datetime
                from influxdb_client import Point

                point = Point(measurement)
                for k, v in tags.items():
                    point = point.tag(k, v)
                for k, v in fields.items():
                    point = point.field(k, v)
                point = point.time(datetime.fromtimestamp(timestamp))

                write_api = self.client.write_api()
                write_api.write(bucket=self.db_config.get("bucket"), record=point)
            else:
                json_body = [
                    {
                        "measurement": measurement,
                        "time": int(timestamp),
                        "tags": tags,
                        "fields": fields,
                    }
                ]
                self.client.write_points(json_body, time_precision="s")

            logger.info("InfluxDB write successful")
            return True

        except Exception as e:
            logger.error("InfluxDB write failed: %s", e)
            self.connected = False
            self.last_error = str(e)
            return False

    # ------------------------------------------------------------------
    # Read queries
    # ------------------------------------------------------------------

    def _query_v1(self, query: str) -> list[dict[str, Any]]:
        result = self.client.query(query)
        rows: list[dict[str, Any]] = []
        for _measurement, points in result.items():
            for point in points:
                rows.append(dict(point))
        return rows

    def query_readings(
        self,
        start: str,
        stop: str,
        rig_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read raw sensor readings by time range.

        Args:
            start: ISO-8601 start time (inclusive).
            stop: ISO-8601 stop time (exclusive).
            rig_id: Optional rig filter.
        """
        if not self._ensure_connected():
            return []

        measurement = self.db_config.get(
            "readings_measurement", "turbidity_readings"
        )
        where = f"time >= '{start}' AND time < '{stop}'"
        if rig_id:
            where += f" AND \"rig_id\" = '{rig_id}'"

        query = f'SELECT * FROM "{measurement}" WHERE {where}'
        try:
            return self._query_v1(query)
        except Exception as e:
            logger.error("query_readings failed: %s", e)
            self.last_error = str(e)
            return []

    def query_predictions(
        self,
        start: str,
        stop: str,
        regime: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read calibrated predictions by time range.

        Args:
            start: ISO-8601 start time (inclusive).
            stop: ISO-8601 stop time (exclusive).
            regime: Optional regime filter.
        """
        if not self._ensure_connected():
            return []

        measurement = self.db_config.get(
            "predictions_measurement", "calibrated_predictions"
        )
        where = f"time >= '{start}' AND time < '{stop}'"
        if regime:
            where += f" AND \"regime\" = '{regime}'"

        query = f'SELECT * FROM "{measurement}" WHERE {where}'
        try:
            return self._query_v1(query)
        except Exception as e:
            logger.error("query_predictions failed: %s", e)
            self.last_error = str(e)
            return []

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("InfluxDB connection closed")


# Global instance (lazy — only connects if enabled in config)
influx_manager = InfluxDBManager()
